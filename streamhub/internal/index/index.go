// Package index keeps an in-memory list of finished segments per camera.
//
// The directory tree is the source of truth: the index is rebuilt by scanning
// it, updated when the recorder finishes a segment, and rescanned periodically
// so files deleted by someone else (pruner, admin) disappear from it.
package index

import (
	"context"
	"errors"
	"io/fs"
	"log/slog"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
)

// Entry is an indexed segment.
type Entry struct {
	segment.Info
	Size int64
}

// Index is safe for concurrent use.
type Index struct {
	root string
	log  *slog.Logger

	mu   sync.RWMutex
	cams map[string][]Entry // sorted by Start
	// While a scan walks the tree, added segments are also collected here and
	// re-applied to its result, so a segment finished mid-scan isn't lost.
	scanning bool
	added    []Entry
}

// New returns an empty index over root.
func New(root string, log *slog.Logger) *Index {
	return &Index{root: root, log: log, cams: map[string][]Entry{}}
}

// Scan rebuilds the index from the directory tree.
func (x *Index) Scan() error {
	x.mu.Lock()
	x.scanning, x.added = true, nil
	x.mu.Unlock()
	defer func() {
		x.mu.Lock()
		x.scanning, x.added = false, nil
		x.mu.Unlock()
	}()

	cams := map[string][]Entry{}
	err := filepath.WalkDir(x.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil // deleted while walking
			}
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, segment.Ext) {
			return nil
		}
		rel, err := filepath.Rel(x.root, path)
		if err != nil {
			return err
		}
		info, err := segment.Parse(rel)
		if err != nil {
			x.log.Debug("ignoring file", "path", rel, "err", err)
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil // deleted while walking
		}
		cams[info.Camera] = append(cams[info.Camera], Entry{Info: info, Size: fi.Size()})
		return nil
	})
	if err != nil {
		return err
	}
	for _, es := range cams {
		sortEntries(es)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	x.cams = cams
	for _, e := range x.added {
		x.insert(e)
	}
	return nil
}

// Add records a newly finished segment.
func (x *Index) Add(info segment.Info, size int64) {
	x.mu.Lock()
	defer x.mu.Unlock()
	e := Entry{Info: info, Size: size}
	if x.scanning {
		x.added = append(x.added, e)
	}
	x.insert(e)
}

// insert adds e unless already present. Caller holds x.mu.
func (x *Index) insert(e Entry) {
	es := x.cams[e.Camera]
	i, found := slices.BinarySearchFunc(es, e.Start, func(a Entry, t time.Time) int { return a.Start.Compare(t) })
	if found && es[i].Path == e.Path {
		return
	}
	x.cams[e.Camera] = slices.Insert(es, i, e)
}

// Remove drops a segment, e.g. after finding its file gone.
func (x *Index) Remove(path string) {
	x.mu.Lock()
	defer x.mu.Unlock()
	for cam, es := range x.cams {
		x.cams[cam] = slices.DeleteFunc(es, func(e Entry) bool { return e.Path == path })
	}
}

// Query returns the camera's segments overlapping [from, to).
func (x *Index) Query(camera string, from, to time.Time) []Entry {
	x.mu.RLock()
	defer x.mu.RUnlock()
	es := x.cams[camera]
	// Segments are short and non-overlapping, so everything starting before
	// `from` except the last one ends before it too; start the search there.
	i, _ := slices.BinarySearchFunc(es, from, func(e Entry, t time.Time) int { return e.Start.Compare(t) })
	if i > 0 {
		i--
	}
	var out []Entry
	for ; i < len(es) && es[i].Start.Before(to); i++ {
		if es[i].End().After(from) {
			out = append(out, es[i])
		}
	}
	return out
}

// Range is a span of continuous recording.
type Range struct {
	Start, End time.Time
}

// Ranges returns the camera's recorded spans overlapping [from, to), merging
// segments separated by at most maxGap.
func (x *Index) Ranges(camera string, from, to time.Time, maxGap time.Duration) []Range {
	var out []Range
	for _, e := range x.Query(camera, from, to) {
		if n := len(out); n > 0 && e.Start.Sub(out[n-1].End) <= maxGap {
			out[n-1].End = e.End()
			continue
		}
		out = append(out, Range{Start: e.Start, End: e.End()})
	}
	return out
}

// RescanEvery rescans the tree periodically until ctx is cancelled.
func (x *Index) RescanEvery(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := x.Scan(); err != nil {
				x.log.Warn("index rescan failed", "err", err)
			}
		}
	}
}

func sortEntries(es []Entry) {
	slices.SortFunc(es, func(a, b Entry) int { return a.Start.Compare(b.Start) })
}
