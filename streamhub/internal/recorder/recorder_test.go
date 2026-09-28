package recorder

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"

	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

// loadAUs splits testdata/tiny.h264 (45 frames, IDR every 10, with access
// unit delimiters) into access units.
func loadAUs(t *testing.T) [][][]byte {
	t.Helper()
	data, err := os.ReadFile("../../testdata/tiny.h264")
	if err != nil {
		t.Fatal(err)
	}
	var nalus [][]byte
	for _, n := range bytes.Split(data, []byte{0, 0, 1}) {
		n = bytes.TrimRight(n, "\x00")
		if len(n) > 0 {
			nalus = append(nalus, n)
		}
	}
	var aus [][][]byte
	for _, n := range nalus {
		if h264.NALUType(n[0]&0x1f) == h264.NALUTypeAccessUnitDelimiter {
			aus = append(aus, nil)
			continue
		}
		aus[len(aus)-1] = append(aus[len(aus)-1], n)
	}
	return aus
}

func paramSets(au [][]byte) (sps, pps []byte) {
	for _, n := range au {
		switch h264.NALUType(n[0] & 0x1f) {
		case h264.NALUTypeSPS:
			sps = n
		case h264.NALUTypePPS:
			pps = n
		}
	}
	return sps, pps
}

var testLog = slog.New(slog.NewTextHandler(io.Discard, nil))

type fed struct {
	pts []int64
}

// feed records AUs with uneven spacing (like a real camera), splitting the
// stream into two sessions at sessionBreak.
func feed(t *testing.T, root string, target time.Duration, sessionBreak int) (fed, []segment.Info) {
	t.Helper()
	aus := loadAUs(t)
	var finished []segment.Info
	rec := New("cam", root, target, func(info segment.Info, _ int64) { finished = append(finished, info) }, testLog)
	stream := media.NewStream()
	sub := stream.Subscribe(len(aus) + 1)

	var f fed
	pts := timeline.TimeToTicks(time.Date(2026, 9, 27, 23, 59, 58, 0, time.UTC))
	var sps, pps []byte
	for i, au := range aus {
		if s, p := paramSets(au); s != nil {
			sps, pps = s, p
		}
		step := int64(6000) // 66.7 ms
		if i%3 == 2 {
			step = 6090 // jitter
		}
		if i == sessionBreak {
			step += 2 * timeline.ClockRate // reconnect gap
		}
		pts += step
		f.pts = append(f.pts, pts)
		stream.Publish(&media.Frame{
			Camera: "cam", PTS: pts, AU: au, IDR: h264.IsRandomAccess(au),
			SPS: sps, PPS: pps, NewSession: i == 0 || i == sessionBreak,
		})
	}
	stream.Unsubscribe(sub)
	if err := rec.Run(context.Background(), sub); err != nil {
		t.Fatal(err)
	}
	return f, finished
}

func probePTS(t *testing.T, path string) (pts []int64, firstKey bool) {
	t.Helper()
	out, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "v",
		"-show_entries", "packet=pts,flags", "-of", "csv=p=0", path).Output()
	if err != nil {
		t.Fatalf("ffprobe %s: %v", path, err)
	}
	lines := strings.Fields(string(out))
	for i, l := range lines {
		ptsStr, flags, _ := strings.Cut(l, ",")
		v, _ := strconv.ParseInt(ptsStr, 10, 64)
		pts = append(pts, v)
		if i == 0 {
			firstKey = strings.HasPrefix(flags, "K")
		}
	}
	return pts, firstKey
}

func TestRecordSegments(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not available")
	}
	root := t.TempDir()
	// Target 1 s at 15 fps with IDR every 10 frames → one GOP (~0.67 s) is not
	// enough, so segments span two GOPs; the session break at frame 25 forces a cut.
	f, finished := feed(t, root, time.Second, 25)

	var all []int64
	for _, info := range finished {
		pts, firstKey := probePTS(t, filepath.Join(root, info.Path))
		if !firstKey {
			t.Errorf("%s does not start with a keyframe", info.Path)
		}
		all = append(all, pts...)
		t.Logf("%s: %d frames", info.Path, len(pts))
	}
	// Frames before the first IDR of the second session are skipped (the
	// session break falls mid-GOP), everything else is recorded with exact PTS.
	var want []int64
	for i, p := range f.pts {
		if i >= 25 && i < 30 {
			continue // after reconnect, waiting for the IDR at frame 30
		}
		want = append(want, p)
	}
	if !slices.Equal(all, want) {
		t.Errorf("recorded PTS differ from fed PTS\n got %v\nwant %v", all, want)
	}
	// Midnight crossing puts segments into different date directories.
	if !strings.HasPrefix(finished[0].Path, "cam/2026-09-27/23/") ||
		!strings.HasPrefix(finished[len(finished)-1].Path, "cam/2026-09-28/00/") {
		t.Errorf("unexpected directories: first %s, last %s", finished[0].Path, finished[len(finished)-1].Path)
	}
	for _, e := range mustGlob(t, root, "*"+segment.PartExt) {
		t.Errorf("leftover part file %s", e)
	}
}

func mustGlob(t *testing.T, root, pattern string) []string {
	t.Helper()
	var out []string
	filepath.WalkDir(root, func(p string, d os.DirEntry, _ error) error {
		if ok, _ := filepath.Match(pattern, d.Name()); ok && !d.IsDir() {
			out = append(out, p)
		}
		return nil
	})
	return out
}

func TestRecoverPartialSegment(t *testing.T) {
	root := t.TempDir()
	_, finished := feed(t, root, time.Hour, -1)
	if len(finished) != 1 {
		t.Fatalf("want 1 segment, got %d", len(finished))
	}
	full := filepath.Join(root, finished[0].Path)
	data, err := os.ReadFile(full)
	if err != nil {
		t.Fatal(err)
	}
	// Simulate a crash: file renamed back to .part and cut mid-fragment.
	start := finished[0].Start
	part := filepath.Join(root, segment.PartPath("cam", start))
	if err := os.WriteFile(part, data[:len(data)-100], 0o644); err != nil {
		t.Fatal(err)
	}
	os.Remove(full)
	// An empty part file (crash before the first fragment) is deleted.
	empty := filepath.Join(root, segment.PartPath("cam", start.Add(time.Minute)))
	os.MkdirAll(filepath.Dir(empty), 0o755)
	os.WriteFile(empty, data[:50], 0o644)

	if err := Recover(root, testLog); err != nil {
		t.Fatal(err)
	}
	if left := mustGlob(t, root, "*"+segment.PartExt); len(left) != 0 {
		t.Fatalf("part files left: %v", left)
	}
	got := mustGlob(t, root, "*"+segment.Ext)
	if len(got) != 1 {
		t.Fatalf("want 1 recovered segment, got %v", got)
	}
	rel, _ := filepath.Rel(root, got[0])
	info, err := segment.Parse(rel)
	if err != nil {
		t.Fatal(err)
	}
	// The last fragment (last GOP, 5 frames) was cut, so the recovered segment is shorter.
	if !info.Start.Equal(start) || info.Duration >= finished[0].Duration || info.Duration < finished[0].Duration/2 {
		t.Errorf("recovered %+v from %+v", info, finished[0])
	}
	if _, err := exec.LookPath("ffprobe"); err == nil {
		if pts, firstKey := probePTS(t, got[0]); !firstKey || len(pts) != 40 {
			t.Errorf("recovered segment: %d frames (want 40), first key %v", len(pts), firstKey)
		}
	}
}
