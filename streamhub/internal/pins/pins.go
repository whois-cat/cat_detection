// Package pins reads and writes pinned time ranges: recordings the pruner must
// keep. The file is <data_dir>/pins.json, a JSON array of Pin.
package pins

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// Pin keeps a camera's recordings (all cameras if Camera is empty) that
// overlap [From, To] (Unix ms).
type Pin struct {
	Camera string `json:"camera,omitempty"`
	From   int64  `json:"from"`
	To     int64  `json:"to"`
	Note   string `json:"note,omitempty"`
}

// Covers reports whether p covers any of camera's [start, end).
func (p Pin) Covers(camera string, start, end time.Time) bool {
	return (p.Camera == "" || p.Camera == camera) &&
		start.UnixMilli() <= p.To && end.UnixMilli() >= p.From
}

// Load reads the pins file; a missing file means no pins.
func Load(path string) ([]Pin, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var ps []Pin
	return ps, json.Unmarshal(b, &ps)
}

// Save writes the pins file atomically.
func Save(path string, ps []Pin) error {
	b, err := json.MarshalIndent(ps, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp")
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
