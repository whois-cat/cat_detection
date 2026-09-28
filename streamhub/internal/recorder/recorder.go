// Package recorder writes a camera's frames to fMP4 segment files (see
// package segment for the layout).
//
// Each segment is a standalone fMP4 file: an init segment followed by
// fragments of one GOP each. Segments start at an IDR frame, end at the first
// IDR after the target length, and also end on reconnect, parameter change,
// or when frames were dropped. Sample timestamps are the frames' wall-clock
// PTS, so segments of a camera form one continuous timeline.
package recorder

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4/seekablebuffer"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/mp4/codecs"

	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

// maxFragment bounds how much video is buffered in memory (and lost on a
// crash) when a camera's GOP is unusually long.
const maxFragment = 5 * time.Second

// FinishFunc is called for every finished segment.
type FinishFunc func(info segment.Info, size int64)

// Recorder records one camera.
type Recorder struct {
	camera   string
	root     string
	target   int64 // ticks
	onFinish FinishFunc
	log      *slog.Logger

	cur     *segWriter
	prev    *media.Frame
	lastDur int64
}

// New returns a Recorder writing segments of roughly target length under root.
func New(camera, root string, target time.Duration, onFinish FinishFunc, log *slog.Logger) *Recorder {
	return &Recorder{
		camera:   camera,
		root:     root,
		target:   int64(target.Seconds() * timeline.ClockRate),
		onFinish: onFinish,
		log:      log.With("camera", camera),
	}
}

// Run records frames from sub until ctx is cancelled or sub is closed, then
// finishes the current segment.
func (r *Recorder) Run(ctx context.Context, sub *media.Subscription) error {
	defer r.flushPrev()
	for {
		select {
		case <-ctx.Done():
			return nil
		case d, ok := <-sub.C:
			if !ok {
				return nil
			}
			if err := r.handle(d); err != nil {
				return err
			}
		}
	}
}

func (r *Recorder) handle(d media.Delivery) error {
	if d.NewSession || d.Gap {
		if d.Gap {
			r.log.Warn("recorder fell behind, frames dropped; starting new segment")
		}
		if err := r.flushPrev(); err != nil {
			return err
		}
	}
	if r.prev != nil {
		if err := r.write(r.prev, d.PTS-r.prev.PTS); err != nil {
			return err
		}
	}
	r.prev = d.Frame
	return nil
}

// flushPrev writes the held-back frame (its duration is unknown, so the
// previous frame's is used) and closes the segment.
func (r *Recorder) flushPrev() error {
	var err error
	if r.prev != nil {
		err = r.write(r.prev, r.lastDur)
		r.prev = nil
	}
	return errors.Join(err, r.closeSegment())
}

func (r *Recorder) write(f *media.Frame, dur int64) error {
	if dur <= 0 {
		dur = 1
	}
	if f.IDR {
		if r.cur != nil && (f.PTS-r.cur.startPTS >= r.target || !r.cur.sameParams(f)) {
			if err := r.closeSegment(); err != nil {
				return err
			}
		}
		if r.cur == nil {
			w, err := openSegment(r.root, r.camera, f)
			if err != nil {
				return err
			}
			r.cur = w
		}
	}
	if r.cur == nil {
		return nil // waiting for an IDR
	}
	r.lastDur = dur
	return r.cur.add(f, dur)
}

func (r *Recorder) closeSegment() error {
	if r.cur == nil {
		return nil
	}
	w := r.cur
	r.cur = nil
	info, size, err := w.finish()
	if err != nil {
		return fmt.Errorf("finish segment %s: %w", w.partPath, err)
	}
	r.onFinish(info, size)
	return nil
}

type segWriter struct {
	root, camera string
	partPath     string // relative to root
	file         *os.File
	sps, pps     []byte
	startPTS     int64
	endPTS       int64
	seq          uint32
	fragBase     int64
	frag         []*fmp4.Sample
}

func openSegment(root, camera string, f *media.Frame) (*segWriter, error) {
	rel := segment.PartPath(camera, timeline.TicksToTime(f.PTS))
	full := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return nil, err
	}
	file, err := os.Create(full)
	if err != nil {
		return nil, err
	}
	w := &segWriter{
		root: root, camera: camera, partPath: rel, file: file,
		sps: f.SPS, pps: f.PPS, startPTS: f.PTS, endPTS: f.PTS,
	}
	init := fmp4.Init{Tracks: []*fmp4.InitTrack{{
		ID:        1,
		TimeScale: timeline.ClockRate,
		Codec:     &codecs.H264{SPS: f.SPS, PPS: f.PPS},
	}}}
	var buf seekablebuffer.Buffer
	if err := init.Marshal(&buf); err != nil {
		file.Close()
		return nil, err
	}
	if _, err := file.Write(buf.Bytes()); err != nil {
		file.Close()
		return nil, err
	}
	return w, nil
}

func (w *segWriter) sameParams(f *media.Frame) bool {
	return string(w.sps) == string(f.SPS) && string(w.pps) == string(f.PPS)
}

func (w *segWriter) add(f *media.Frame, dur int64) error {
	if len(w.frag) > 0 && (f.IDR || f.PTS-w.fragBase >= int64(maxFragment.Seconds()*timeline.ClockRate)) {
		if err := w.flushFragment(); err != nil {
			return err
		}
	}
	if len(w.frag) == 0 {
		w.fragBase = f.PTS
	}
	payload, err := h264.AVCC(f.AU).Marshal()
	if err != nil {
		return err
	}
	w.frag = append(w.frag, &fmp4.Sample{
		Duration:        uint32(dur),
		IsNonSyncSample: !f.IDR,
		Payload:         payload,
	})
	w.endPTS = f.PTS + dur
	return nil
}

func (w *segWriter) flushFragment() error {
	if len(w.frag) == 0 {
		return nil
	}
	part := fmp4.Part{SequenceNumber: w.seq, Tracks: []*fmp4.PartTrack{{
		ID: 1, BaseTime: uint64(w.fragBase), Samples: w.frag,
	}}}
	w.seq++
	w.frag = nil
	var buf seekablebuffer.Buffer
	if err := part.Marshal(&buf); err != nil {
		return err
	}
	_, err := w.file.Write(buf.Bytes())
	return err
}

func (w *segWriter) finish() (segment.Info, int64, error) {
	err := errors.Join(w.flushFragment(), w.file.Close())
	if err != nil {
		return segment.Info{}, 0, err
	}
	return finalize(w.root, w.camera, w.partPath, w.startPTS, w.endPTS)
}

// finalize renames a .part file to its final name.
func finalize(root, camera, partPath string, startPTS, endPTS int64) (segment.Info, int64, error) {
	start := timeline.TicksToTime(startPTS)
	dur := timeline.TicksToDuration(endPTS - startPTS)
	final := segment.FinalPath(camera, start, dur)
	if err := os.Rename(filepath.Join(root, partPath), filepath.Join(root, final)); err != nil {
		return segment.Info{}, 0, err
	}
	st, err := os.Stat(filepath.Join(root, final))
	if err != nil {
		return segment.Info{}, 0, err
	}
	info, err := segment.Parse(final)
	return info, st.Size(), err
}

// Recover finalizes .part files left by an unclean shutdown: truncates each to
// its last complete fragment and renames it, or deletes it if it holds no
// complete fragment. Must run before recording starts.
func Recover(root string, log *slog.Logger) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, segment.PartExt) {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		camera := strings.SplitN(filepath.ToSlash(rel), "/", 2)[0]
		info, err := recoverPart(root, camera, rel)
		if err != nil {
			log.Warn("deleting unrecoverable partial segment", "path", rel, "err", err)
			return os.Remove(path)
		}
		log.Info("recovered partial segment", "path", info.Path, "duration", info.Duration)
		return nil
	})
}

func recoverPart(root, camera, rel string) (segment.Info, error) {
	full := filepath.Join(root, rel)
	data, err := os.ReadFile(full)
	if err != nil {
		return segment.Info{}, err
	}
	end := completeLength(data)
	var parts fmp4.Parts
	if err := parts.Unmarshal(data[:end]); err != nil {
		return segment.Info{}, err
	}
	if len(parts) == 0 {
		return segment.Info{}, errors.New("no complete fragment")
	}
	first := parts[0].Tracks[0]
	last := parts[len(parts)-1].Tracks[0]
	endPTS := int64(last.BaseTime)
	for _, s := range last.Samples {
		endPTS += int64(s.Duration)
	}
	if err := os.Truncate(full, int64(end)); err != nil {
		return segment.Info{}, err
	}
	info, _, err := finalize(root, camera, rel, int64(first.BaseTime), endPTS)
	return info, err
}

// completeLength returns the length of data up to the end of the last
// complete mdat box (i.e. the last complete fragment), or up to the last
// complete box if there is no fragment.
func completeLength(data []byte) int {
	pos, lastMdatEnd := 0, 0
	for pos+8 <= len(data) {
		size := int(binary.BigEndian.Uint32(data[pos:]))
		typ := string(data[pos+4 : pos+8])
		if size == 1 && pos+16 <= len(data) {
			size = int(binary.BigEndian.Uint64(data[pos+8:]))
		}
		if size < 8 || pos+size > len(data) {
			break
		}
		pos += size
		if typ == "mdat" {
			lastMdatEnd = pos
		}
	}
	if lastMdatEnd > 0 {
		return lastMdatEnd
	}
	return pos
}
