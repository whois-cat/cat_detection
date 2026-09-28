// Package sidecar stores CV results next to recorded segments and summarizes
// them for the timeline.
//
// Each segment has a sidecar <start>.labels.jsonl (see package segment) with
// one JSON object per line, typed by "t":
//
//	{"t":"cv","pts":…,"model":…,"worker":…,"infer_ms":…,"dets":[…]}
//	{"t":"decision","pts":…,"feeder":…,"state":…,"reason":…,…}
//
// written as they arrive. CV results are written for every processed frame
// (so "looked, found nothing" is distinguishable from "didn't look");
// decisions when a decider's state changes, and each feeder's latest decision
// again at the start of every segment (with its original pts), so any segment
// on its own shows the state in effect.
package sidecar

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

// Line types.
const (
	TypeCV       = "cv"
	TypeDecision = "decision"
)

// recentStarts is how many segment starts per camera are remembered for
// routing late results.
const recentStarts = 64

// Line is one sidecar line.
type Line struct {
	T string `json:"t"`
	labels.Result
}

// Writer appends results to the sidecar of the segment containing their frame.
type Writer struct {
	root string
	sum  *Summary
	log  *slog.Logger

	mu     sync.Mutex
	starts map[string][]int64                    // camera -> recent segment start PTS, ascending
	last   map[string]map[string]labels.Decision // camera -> feeder -> latest decision
}

// NewWriter returns a Writer under the recordings root.
func NewWriter(root string, sum *Summary, log *slog.Logger) *Writer {
	return &Writer{root: root, sum: sum, log: log, starts: map[string][]int64{}, last: map[string]map[string]labels.Decision{}}
}

// SegmentStarted tells the writer that camera's recorder opened a segment
// whose first frame has startPTS.
func (w *Writer) SegmentStarted(camera string, startPTS int64) {
	w.mu.Lock()
	s := append(w.starts[camera], startPTS)
	if len(s) > recentStarts {
		s = s[len(s)-recentStarts:]
	}
	w.starts[camera] = s
	carried := slices.Collect(maps.Values(w.last[camera]))
	w.mu.Unlock()
	rel := filepath.ToSlash(segment.SidecarPath(camera, timeline.TicksToTime(startPTS)))
	for _, d := range carried {
		if err := w.appendTo(rel, decisionLine(d)); err != nil {
			w.log.Warn("carrying decision into new sidecar", "camera", camera, "err", err)
		}
	}
}

// segmentOf returns the start PTS of the segment holding pts.
func (w *Writer) segmentOf(camera string, pts int64) (int64, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	s := w.starts[camera]
	i, found := slices.BinarySearch(s, pts)
	if found {
		return s[i], true
	}
	if i == 0 {
		return 0, false // before the first known segment (e.g. not recorded)
	}
	return s[i-1], true
}

// Run writes results and decisions until ctx is done.
func (w *Writer) Run(ctx context.Context, results <-chan labels.Result, decisions <-chan labels.Decision) {
	for {
		var err error
		var camera string
		select {
		case <-ctx.Done():
			return
		case r := <-results:
			camera, err = r.Camera, w.Write(r)
		case d := <-decisions:
			camera, err = d.Camera, w.WriteDecision(d)
		}
		if err != nil {
			w.log.Warn("writing sidecar", "camera", camera, "err", err)
		}
	}
}

// Write stores one result.
func (w *Writer) Write(r labels.Result) error {
	line, err := json.Marshal(Line{T: TypeCV, Result: r})
	if err != nil {
		return err
	}
	// Summarize before writing so a concurrent Summary.Sync never loads this line too.
	return w.append(r.Camera, r.PTS, line, func(rel string) { w.sum.Add(rel, r) })
}

// WriteDecision stores one decision.
func (w *Writer) WriteDecision(d labels.Decision) error {
	feeder, _ := d.Fields["feeder"].(string)
	w.mu.Lock()
	if w.last[d.Camera] == nil {
		w.last[d.Camera] = map[string]labels.Decision{}
	}
	w.last[d.Camera][feeder] = d
	w.mu.Unlock()
	return w.append(d.Camera, d.PTS, decisionLine(d), nil)
}

func decisionLine(d labels.Decision) []byte {
	m := maps.Clone(d.Fields)
	m["t"] = TypeDecision
	line, _ := json.Marshal(m) // Fields came from msgpack: always encodable
	return line
}

// append adds a line to the sidecar of the segment holding pts.
func (w *Writer) append(camera string, pts int64, line []byte, before func(rel string)) error {
	start, ok := w.segmentOf(camera, pts)
	if !ok {
		return nil
	}
	rel := filepath.ToSlash(segment.SidecarPath(camera, timeline.TicksToTime(start)))
	if before != nil {
		before(rel)
	}
	return w.appendTo(rel, line)
}

func (w *Writer) appendTo(rel string, line []byte) error {
	full := filepath.Join(w.root, filepath.FromSlash(rel))
	f, err := os.OpenFile(full, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	_, err = f.Write(append(line, '\n'))
	return errors.Join(err, f.Close())
}

// Bucket counts detections of one cat within one second.
type Bucket struct {
	Sec int64  // Unix seconds
	Cat string // labels.Det.TopCat
	N   int
}

type fileSum struct {
	camera string
	start  time.Time
	secs   map[int64]map[string]int
}

// maxSegment bounds how long a segment can be, for range queries by start time.
const maxSegment = 10 * time.Minute

// Summary keeps per-second detection counts per sidecar file. Files are the
// source of truth: Sync picks up new ones and forgets deleted ones.
type Summary struct {
	root string
	log  *slog.Logger

	mu    sync.RWMutex
	files map[string]*fileSum // rel path -> summary
}

// NewSummary returns an empty Summary over the recordings root.
func NewSummary(root string, log *slog.Logger) *Summary {
	return &Summary{root: root, log: log, files: map[string]*fileSum{}}
}

// Add counts a result belonging to sidecar rel.
func (s *Summary) Add(rel string, r labels.Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fs := s.files[rel]
	if fs == nil {
		camera, start, err := segment.ParseSidecar(rel)
		if err != nil {
			return
		}
		fs = &fileSum{camera: camera, start: start, secs: map[int64]map[string]int{}}
		s.files[rel] = fs
	}
	fs.add(r)
}

func (f *fileSum) add(r labels.Result) {
	if len(r.Dets) == 0 {
		return
	}
	sec := timeline.TicksToTime(r.PTS).Unix()
	m := f.secs[sec]
	if m == nil {
		m = map[string]int{}
		f.secs[sec] = m
	}
	for _, d := range r.Dets {
		m[d.TopCat()]++
	}
}

// Sync loads sidecars not yet known and forgets deleted ones.
func (s *Summary) Sync() error {
	seen := map[string]bool{}
	err := filepath.WalkDir(s.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, segment.SidecarExt) {
			return nil
		}
		rel, err := filepath.Rel(s.root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		seen[rel] = true
		s.mu.RLock()
		known := s.files[rel] != nil
		s.mu.RUnlock()
		if !known {
			if err := s.load(rel); err != nil {
				s.log.Warn("reading sidecar", "path", rel, "err", err)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for rel := range s.files {
		if !seen[rel] {
			delete(s.files, rel)
		}
	}
	return nil
}

func (s *Summary) load(rel string) error {
	camera, start, err := segment.ParseSidecar(rel)
	if err != nil {
		return err
	}
	f, err := os.Open(filepath.Join(s.root, filepath.FromSlash(rel)))
	if err != nil {
		return err
	}
	defer f.Close()
	fsum := &fileSum{camera: camera, start: start, secs: map[int64]map[string]int{}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		var l Line
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil || l.T != TypeCV {
			continue // partial last line or another line type
		}
		fsum.add(l.Result)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.files[rel] == nil { // Add may have created it meanwhile
		s.files[rel] = fsum
	}
	return sc.Err()
}

// Query returns camera's detection buckets within [from, to), sorted by time.
func (s *Summary) Query(camera string, from, to time.Time) []Bucket {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []Bucket
	for _, f := range s.files {
		if f.camera != camera || f.start.Before(from.Add(-maxSegment)) || !f.start.Before(to) {
			continue
		}
		for sec, cats := range f.secs {
			if t := time.Unix(sec, 0); t.Before(from.Truncate(time.Second)) || !t.Before(to) {
				continue
			}
			for cat, n := range cats {
				out = append(out, Bucket{Sec: sec, Cat: cat, N: n})
			}
		}
	}
	slices.SortFunc(out, func(a, b Bucket) int {
		if a.Sec != b.Sec {
			return int(a.Sec - b.Sec)
		}
		return strings.Compare(a.Cat, b.Cat)
	})
	return out
}

// SyncEvery calls Sync periodically until ctx is done.
func (s *Summary) SyncEvery(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := s.Sync(); err != nil {
				s.log.Warn("sidecar summary sync failed", "err", err)
			}
		}
	}
}
