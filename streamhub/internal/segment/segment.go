// Package segment defines the on-disk layout of recordings, shared by
// streamhub and pruner:
//
//	<root>/<camera>/<YYYY-MM-DD>/<HH>/<start>_<duration>.mp4   finished segment
//	<root>/<camera>/<YYYY-MM-DD>/<HH>/<start>.part             segment being written
//
// <start> is the UTC wall-clock time of the first frame with millisecond
// precision (e.g. 2026-09-27T23-55-18.123Z), <duration> is in milliseconds
// (e.g. 10033ms). Date and hour directories are those of <start>. Everything
// needed to index a finished segment is in its path.
package segment

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	stampLayout = "2006-01-02T15-04-05.000Z"
	dateLayout  = "2006-01-02"
	hourLayout  = "15"

	// Ext is the extension of finished segments.
	Ext = ".mp4"
	// PartExt is the extension of segments still being written.
	PartExt = ".part"
	// SidecarExt is the extension of a segment's label sidecar
	// (<start>.labels.jsonl): named by start time only, so it pairs with the
	// segment both while it's a .part and once finished.
	SidecarExt = ".labels.jsonl"
)

// Info describes a finished segment.
type Info struct {
	Camera   string
	Start    time.Time
	Duration time.Duration
	// Path is relative to the recordings root, with forward slashes.
	Path string
}

// End returns the end time of the segment.
func (i Info) End() time.Time { return i.Start.Add(i.Duration) }

func dir(camera string, start time.Time) string {
	start = start.UTC()
	return filepath.Join(camera, start.Format(dateLayout), start.Format(hourLayout))
}

// PartPath returns the relative path of the in-progress file for a segment
// starting at start.
func PartPath(camera string, start time.Time) string {
	return filepath.Join(dir(camera, start), start.UTC().Format(stampLayout)+PartExt)
}

// FinalPath returns the relative path of a finished segment.
func FinalPath(camera string, start time.Time, duration time.Duration) string {
	return filepath.Join(dir(camera, start), fmt.Sprintf("%s_%dms%s",
		start.UTC().Format(stampLayout), duration.Milliseconds(), Ext))
}

// SidecarPath returns the relative path of the label sidecar of the segment
// starting at start.
func SidecarPath(camera string, start time.Time) string {
	return filepath.Join(dir(camera, start), start.UTC().Format(stampLayout)+SidecarExt)
}

// Sidecar returns the relative path (forward slashes) of i's label sidecar.
func (i Info) Sidecar() string { return filepath.ToSlash(SidecarPath(i.Camera, i.Start)) }

// ParseSidecar parses the relative path of a label sidecar.
func ParseSidecar(relPath string) (camera string, start time.Time, err error) {
	relPath = filepath.ToSlash(relPath)
	parts := strings.Split(relPath, "/")
	if len(parts) != 4 {
		return "", time.Time{}, fmt.Errorf("sidecar path %q: want <camera>/<date>/<hour>/<file>", relPath)
	}
	stamp, ok := strings.CutSuffix(parts[3], SidecarExt)
	if !ok {
		return "", time.Time{}, fmt.Errorf("sidecar path %q: not a %s file", relPath, SidecarExt)
	}
	start, err = time.Parse(stampLayout, stamp)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("sidecar path %q: %w", relPath, err)
	}
	if want := filepath.ToSlash(SidecarPath(parts[0], start)); want != relPath {
		return "", time.Time{}, fmt.Errorf("sidecar path %q: directory does not match start time", relPath)
	}
	return parts[0], start, nil
}

// Parse parses the relative path of a finished segment.
func Parse(relPath string) (Info, error) {
	relPath = filepath.ToSlash(relPath)
	parts := strings.Split(relPath, "/")
	if len(parts) != 4 {
		return Info{}, fmt.Errorf("segment path %q: want <camera>/<date>/<hour>/<file>", relPath)
	}
	name, ok := strings.CutSuffix(parts[3], Ext)
	if !ok {
		return Info{}, fmt.Errorf("segment path %q: not a %s file", relPath, Ext)
	}
	stamp, dur, ok := strings.Cut(name, "_")
	if !ok {
		return Info{}, fmt.Errorf("segment path %q: missing duration", relPath)
	}
	start, err := time.Parse(stampLayout, stamp)
	if err != nil {
		return Info{}, fmt.Errorf("segment path %q: %w", relPath, err)
	}
	msStr, ok := strings.CutSuffix(dur, "ms")
	if !ok {
		return Info{}, fmt.Errorf("segment path %q: bad duration %q", relPath, dur)
	}
	ms, err := strconv.ParseInt(msStr, 10, 64)
	if err != nil || ms < 0 {
		return Info{}, fmt.Errorf("segment path %q: bad duration %q", relPath, dur)
	}
	info := Info{Camera: parts[0], Start: start, Duration: time.Duration(ms) * time.Millisecond, Path: relPath}
	if want := filepath.ToSlash(FinalPath(info.Camera, info.Start, info.Duration)); want != relPath {
		return Info{}, fmt.Errorf("segment path %q: directory does not match start time (want %q)", relPath, want)
	}
	return info, nil
}
