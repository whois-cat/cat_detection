// Package prune decides which recordings to delete (see cmd/pruner).
//
// Files are the source of truth: each pass lists segments and their sidecars
// from disk, so it works no matter who else deletes files.
package prune

import (
	"bufio"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/pins"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/sidecar"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

// Policy is what to keep.
type Policy struct {
	// KeepRecent: segments ending less than this ago are never sparsified.
	KeepRecent time.Duration
	// EventMargin: older segments are kept if a detection is within this
	// distance of them.
	EventMargin time.Duration
	// DeleteUnprocessed deletes old segments CV never looked at (no sidecar).
	// Otherwise they are kept (subject to MaxBytes).
	DeleteUnprocessed bool
	// MaxBytes caps the total size of recordings; oldest unpinned go first.
	MaxBytes int64
}

// Reason tells why a segment is deleted.
type Reason string

const (
	NoEvents    Reason = "no detections nearby"
	Unprocessed Reason = "never processed by CV"
	OverSize    Reason = "over size cap"
)

// Deletion is one planned deletion.
type Deletion struct {
	Seg    Segment
	Reason Reason
}

// Segment is a finished segment on disk.
type Segment struct {
	segment.Info
	Bytes      int64 // segment + sidecar
	HasSidecar bool
	// Detections are the PTS (as time) of results with detections.
	Detections []time.Time
}

// Scan lists finished segments with their sidecars, per camera sorted by start.
func Scan(root string) (map[string][]Segment, error) {
	cams := map[string][]Segment{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, segment.Ext) {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		info, err := segment.Parse(rel)
		if err != nil {
			return nil // not ours
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		s := Segment{Info: info, Bytes: fi.Size()}
		if sfi, err := os.Stat(filepath.Join(root, filepath.FromSlash(info.Sidecar()))); err == nil {
			s.HasSidecar = true
			s.Bytes += sfi.Size()
			s.Detections, err = detections(filepath.Join(root, filepath.FromSlash(info.Sidecar())))
			if err != nil {
				return err
			}
		}
		cams[info.Camera] = append(cams[info.Camera], s)
		return nil
	})
	for _, ss := range cams {
		slices.SortFunc(ss, func(a, b Segment) int { return a.Start.Compare(b.Start) })
	}
	return cams, err
}

func detections(path string) ([]time.Time, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []time.Time
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		var l sidecar.Line
		if json.Unmarshal(sc.Bytes(), &l) == nil && l.T == sidecar.TypeCV && len(l.Dets) > 0 {
			out = append(out, timeline.TicksToTime(l.PTS))
		}
	}
	return out, sc.Err()
}

// Plan decides what to delete.
func Plan(cams map[string][]Segment, ps []pins.Pin, p Policy, now time.Time) []Deletion {
	pinned := func(s Segment) bool {
		return slices.ContainsFunc(ps, func(pin pins.Pin) bool { return pin.Covers(s.Camera, s.Start, s.End()) })
	}
	var dels []Deletion
	deleted := map[string]bool{}
	for _, ss := range cams {
		var events []time.Time
		for _, s := range ss {
			events = append(events, s.Detections...)
		}
		slices.SortFunc(events, time.Time.Compare)
		for _, s := range ss {
			if now.Sub(s.End()) < p.KeepRecent || pinned(s) {
				continue
			}
			if !s.HasSidecar {
				if p.DeleteUnprocessed {
					dels = append(dels, Deletion{s, Unprocessed})
					deleted[s.Path] = true
				}
				continue
			}
			// Any detection in [start - margin, end + margin]?
			i, _ := slices.BinarySearchFunc(events, s.Start.Add(-p.EventMargin), time.Time.Compare)
			if i < len(events) && !events[i].After(s.End().Add(p.EventMargin)) {
				continue
			}
			dels = append(dels, Deletion{s, NoEvents})
			deleted[s.Path] = true
		}
	}

	if p.MaxBytes > 0 {
		var remaining []Segment
		var total int64
		for _, ss := range cams {
			for _, s := range ss {
				if !deleted[s.Path] {
					remaining = append(remaining, s)
					total += s.Bytes
				}
			}
		}
		slices.SortFunc(remaining, func(a, b Segment) int { return a.Start.Compare(b.Start) })
		for _, s := range remaining {
			if total <= p.MaxBytes {
				break
			}
			if pinned(s) {
				continue
			}
			dels = append(dels, Deletion{s, OverSize})
			total -= s.Bytes
		}
	}
	return dels
}

// Apply deletes planned segments with their sidecars and removes directories
// left empty. With dryRun it only logs.
func Apply(root string, dels []Deletion, dryRun bool, log *slog.Logger) (freed int64, err error) {
	dirs := map[string]bool{}
	for _, d := range dels {
		log.Info("deleting segment", "path", d.Seg.Path, "reason", d.Reason, "bytes", d.Seg.Bytes, "dry_run", dryRun)
		if dryRun {
			continue
		}
		for _, rel := range []string{d.Seg.Path, d.Seg.Sidecar()} {
			if e := os.Remove(filepath.Join(root, filepath.FromSlash(rel))); e != nil && !errors.Is(e, fs.ErrNotExist) {
				err = errors.Join(err, e)
			}
		}
		freed += d.Seg.Bytes
		dirs[filepath.Dir(filepath.Join(root, filepath.FromSlash(d.Seg.Path)))] = true
	}
	for dir := range dirs {
		// Hour dir, then date dir; Remove fails harmlessly while non-empty.
		for range 2 {
			if os.Remove(dir) != nil {
				break
			}
			dir = filepath.Dir(dir)
		}
	}
	return freed, err
}
