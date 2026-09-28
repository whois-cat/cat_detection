package sidecar

import (
	"bufio"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

var testLog = slog.New(slog.NewTextHandler(io.Discard, nil))

func TestWriteRouteAndSummarize(t *testing.T) {
	root := t.TempDir()
	sum := NewSummary(root, testLog)
	w := NewWriter(root, sum, testLog)
	t0 := time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)
	ticks := func(d time.Duration) int64 { return timeline.TimeToTicks(t0.Add(d)) }
	for _, d := range []time.Duration{0, 10 * time.Second} {
		start := t0.Add(d)
		os.MkdirAll(filepath.Join(root, filepath.Dir(segment.SidecarPath("grey", start))), 0o755)
		w.SegmentStarted("grey", ticks(d))
	}
	cat := func(p float64) labels.Det {
		return labels.Det{Box: [4]float64{0.1, 0.1, 0.2, 0.2}, Score: 0.9, Cats: map[string]float64{"alisa": p, "chuzh": 1 - p}}
	}
	unsure := labels.Det{Score: 0.8, Cats: map[string]float64{"alisa": 0.4, "chuzh": 0.35, "felisis": 0.25}}
	results := []labels.Result{
		{Camera: "grey", PTS: ticks(-time.Second)},                                                 // before any segment: dropped
		{Camera: "grey", PTS: ticks(500 * time.Millisecond), Dets: []labels.Det{cat(0.9)}},         // segment 1
		{Camera: "grey", PTS: ticks(700 * time.Millisecond), Dets: []labels.Det{cat(0.9), unsure}}, // same second
		{Camera: "grey", PTS: ticks(3 * time.Second)},                                              // nothing found
		{Camera: "grey", PTS: ticks(12 * time.Second), Dets: []labels.Det{cat(0.3)}},               // segment 2
	}
	for _, r := range results {
		if err := w.Write(r); err != nil {
			t.Fatal(err)
		}
	}

	lines := readLines(t, filepath.Join(root, segment.SidecarPath("grey", t0)))
	if len(lines) != 3 || lines[2].PTS != ticks(3*time.Second) || len(lines[2].Dets) != 0 {
		t.Fatalf("segment 1 sidecar: %+v", lines)
	}
	if lines := readLines(t, filepath.Join(root, segment.SidecarPath("grey", t0.Add(10*time.Second)))); len(lines) != 1 {
		t.Fatalf("segment 2 sidecar: %+v", lines)
	}

	want := []Bucket{
		{Sec: t0.Unix(), Cat: "alisa", N: 2},
		{Sec: t0.Unix(), Cat: "unknown", N: 1},
		{Sec: t0.Unix() + 12, Cat: "chuzh", N: 1},
	}
	check := func(what string, s *Summary) {
		t.Helper()
		got := s.Query("grey", t0.Add(-time.Minute), t0.Add(time.Minute))
		if len(got) != len(want) {
			t.Fatalf("%s: got %+v", what, got)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: got %+v, want %+v", what, got, want)
			}
		}
	}
	check("live summary", sum)

	// A fresh summary rebuilt from the files agrees; a deleted file is forgotten.
	fresh := NewSummary(root, testLog)
	if err := fresh.Sync(); err != nil {
		t.Fatal(err)
	}
	check("rebuilt summary", fresh)
	os.Remove(filepath.Join(root, segment.SidecarPath("grey", t0.Add(10*time.Second))))
	sum.Sync()
	if got := sum.Query("grey", t0.Add(10*time.Second), t0.Add(time.Minute)); len(got) != 0 {
		t.Fatalf("deleted sidecar still summarized: %+v", got)
	}
}

func readLines(t *testing.T, path string) []Line {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []Line
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var l Line
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil || l.T != TypeCV {
			t.Fatalf("bad line %q: %v", sc.Text(), err)
		}
		out = append(out, l)
	}
	return out
}
