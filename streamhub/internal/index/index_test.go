package index

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
)

var testLog = slog.New(slog.NewTextHandler(io.Discard, nil))

func touch(t *testing.T, root, rel string) {
	t.Helper()
	full := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScanQueryRemove(t *testing.T) {
	root := t.TempDir()
	t0 := time.Date(2026, 9, 27, 23, 59, 50, 0, time.UTC)
	var paths []string
	for i := range 3 {
		p := segment.FinalPath("grey", t0.Add(time.Duration(i)*10*time.Second), 10*time.Second)
		touch(t, root, p)
		paths = append(paths, p)
	}
	touch(t, root, segment.FinalPath("beige", t0, 10*time.Second))
	touch(t, root, segment.PartPath("grey", t0.Add(30*time.Second))) // in progress: not indexed
	touch(t, root, "grey/README.txt")                                // foreign file: ignored

	x := New(root, testLog)
	if err := x.Scan(); err != nil {
		t.Fatal(err)
	}
	if got := x.Query("grey", t0, t0.Add(time.Hour)); len(got) != 3 {
		t.Fatalf("query all: got %d", len(got))
	}
	// Overlap semantics: [t0+15s, t0+16s) touches only the second segment.
	got := x.Query("grey", t0.Add(15*time.Second), t0.Add(16*time.Second))
	if len(got) != 1 || got[0].Path != filepath.ToSlash(paths[1]) {
		t.Fatalf("overlap query: %+v", got)
	}
	if got := x.Query("grey", t0.Add(30*time.Second), t0.Add(time.Hour)); len(got) != 0 {
		t.Fatalf("query past end: %+v", got)
	}

	// External deletion is picked up by a rescan.
	os.Remove(filepath.Join(root, paths[0]))
	if err := x.Scan(); err != nil {
		t.Fatal(err)
	}
	if got := x.Query("grey", t0, t0.Add(time.Hour)); len(got) != 2 {
		t.Fatalf("after delete: got %d", len(got))
	}

	x.Remove(filepath.ToSlash(paths[1]))
	x.Add(segment.Info{Camera: "grey", Start: t0.Add(-time.Minute), Duration: 10 * time.Second, Path: "p"}, 1)
	got = x.Query("grey", t0.Add(-time.Hour), t0.Add(time.Hour))
	if len(got) != 2 || got[0].Path != "p" {
		t.Fatalf("after remove/add: %+v", got)
	}
}
