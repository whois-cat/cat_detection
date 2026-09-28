package prune

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/pins"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/sidecar"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

var (
	testLog = slog.New(slog.NewTextHandler(io.Discard, nil))
	t0      = time.Date(2026, 9, 27, 10, 0, 0, 0, time.UTC)
	now     = t0.Add(24 * time.Hour)
)

// seg writes a 10 s segment of 1000 bytes starting at t0+at. sidecar: nil =
// none; otherwise results at the given offsets into the segment, with a
// detection when the offset is positive.
func seg(t *testing.T, root, cam string, at time.Duration, results ...time.Duration) string {
	t.Helper()
	start := t0.Add(at)
	rel := segment.FinalPath(cam, start, 10*time.Second)
	full := filepath.Join(root, rel)
	os.MkdirAll(filepath.Dir(full), 0o755)
	os.WriteFile(full, make([]byte, 1000), 0o644)
	if results != nil {
		f, _ := os.Create(filepath.Join(root, segment.SidecarPath(cam, start)))
		for _, off := range results {
			r := labels.Result{PTS: timeline.TimeToTicks(start.Add(off.Abs()))}
			if off > 0 {
				r.Dets = []labels.Det{{Score: 0.9}}
			}
			b, _ := json.Marshal(sidecar.Line{T: sidecar.TypeCV, Result: r})
			f.Write(append(b, '\n'))
		}
		f.Close()
	}
	return filepath.ToSlash(rel)
}

func TestPlanAndApply(t *testing.T) {
	root := t.TempDir()
	nothing := []time.Duration{-time.Second} // processed, no detection
	var (
		quiet     = seg(t, root, "grey", 0, nothing...)
		beforeCat = seg(t, root, "grey", 10*time.Minute, nothing...) // ends 20 s before the detection
		withCat   = seg(t, root, "grey", 10*time.Minute+30*time.Second, 5*time.Second)
		afterCat  = seg(t, root, "grey", 10*time.Minute+40*time.Second, nothing...) // starts 5 s after it... within margin
		farAfter  = seg(t, root, "grey", 11*time.Minute+30*time.Second, nothing...)
		noCV      = seg(t, root, "grey", time.Hour)
		pinned    = seg(t, root, "grey", 2*time.Hour, nothing...)
		recent    = seg(t, root, "grey", 23*time.Hour, nothing...)
		otherCam  = seg(t, root, "beige", 10*time.Minute+30*time.Second, nothing...) // grey's cat doesn't count
	)
	cams, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	ps := []pins.Pin{{Camera: "grey", From: t0.Add(2 * time.Hour).UnixMilli(), To: t0.Add(2*time.Hour + time.Second).UnixMilli()}}
	policy := Policy{KeepRecent: 3 * time.Hour, EventMargin: 30 * time.Second, DeleteUnprocessed: true}

	got := map[string]Reason{}
	for _, d := range Plan(cams, ps, policy, now) {
		got[d.Seg.Path] = d.Reason
	}
	want := map[string]Reason{quiet: NoEvents, farAfter: NoEvents, noCV: Unprocessed, otherCam: NoEvents}
	if len(got) != len(want) {
		t.Errorf("plan = %v, want %v", got, want)
	}
	for p, r := range want {
		if got[p] != r {
			t.Errorf("%s: %q, want %q", p, got[p], r)
		}
	}
	for _, kept := range []string{beforeCat, withCat, afterCat, pinned, recent} {
		if _, ok := got[kept]; ok {
			t.Errorf("%s should be kept", kept)
		}
	}

	policy.DeleteUnprocessed = false
	for _, d := range Plan(cams, ps, policy, now) {
		if d.Seg.Path == noCV {
			t.Error("unprocessed segment deleted with DeleteUnprocessed off")
		}
	}

	// Size cap: 9 segments; the sidecar-less one is 1000 bytes, the rest a bit more.
	// Keep ~3 segments' worth: oldest unpinned go first, recent ones too if needed.
	policy.MaxBytes = 3500
	dels := Plan(cams, ps, policy, now)
	var capped []string
	for _, d := range dels {
		if d.Reason == OverSize {
			capped = append(capped, d.Seg.Path)
		}
	}
	if !slices.Contains(capped, beforeCat) || slices.Contains(capped, pinned) || slices.Contains(capped, recent) {
		t.Errorf("size cap deleted %v", capped)
	}

	freed, err := Apply(root, dels, false, testLog)
	if err != nil || freed == 0 {
		t.Fatalf("apply: freed %d, err %v", freed, err)
	}
	after, _ := Scan(root)
	var left []string
	for _, ss := range after {
		for _, s := range ss {
			left = append(left, s.Path)
		}
	}
	slices.Sort(left)
	wantLeft := []string{pinned, recent, noCV}
	slices.Sort(wantLeft)
	if !slices.Equal(left, wantLeft) {
		t.Errorf("left %v, want %v", left, wantLeft)
	}
	// Sidecars went with their segments; empty dirs are gone.
	if _, err := os.Stat(filepath.Join(root, segment.SidecarPath("grey", t0))); !os.IsNotExist(err) {
		t.Error("sidecar of deleted segment still exists")
	}
	if _, err := os.Stat(filepath.Join(root, "beige")); err != nil {
		t.Error("camera dir should stay")
	}
	if entries, _ := os.ReadDir(filepath.Join(root, "beige")); len(entries) != 0 {
		t.Errorf("empty date dir not removed: %v", entries)
	}
}

func TestDryRun(t *testing.T) {
	root := t.TempDir()
	p := seg(t, root, "grey", 0, -time.Second)
	cams, _ := Scan(root)
	dels := Plan(cams, nil, Policy{KeepRecent: time.Hour}, now)
	if len(dels) != 1 {
		t.Fatalf("plan: %v", dels)
	}
	if _, err := Apply(root, dels, true, testLog); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, p)); err != nil {
		t.Error("dry run deleted a file")
	}
}
