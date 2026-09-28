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
}

// New returns an empty index over root.
func New(root string, log *slog.Logger) *Index {
	return &Index{root: root, log: log, cams: map[string][]Entry{}}
}

// Scan rebuilds the index from the directory tree.
func (x *Index) Scan() error {
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
	x.cams = cams
	x.mu.Unlock()
	return nil
}

// Add records a newly finished segment.
func (x *Index) Add(info segment.Info, size int64) {
	x.mu.Lock()
	defer x.mu.Unlock()
	es := x.cams[info.Camera]
	i, _ := slices.BinarySearchFunc(es, info.Start, func(e Entry, t time.Time) int { return e.Start.Compare(t) })
	x.cams[info.Camera] = slices.Insert(es, i, Entry{Info: info, Size: size})
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
