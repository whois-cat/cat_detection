// Package api serves streamhub's HTTP API.
package api

import (
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/index"
	"github.com/whois-cat/cat_detection/streamhub/internal/ingest"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
)

// defaultRange is the recordings query window when from/to are omitted.
const defaultRange = time.Hour

// Server holds what the handlers need.
type Server struct {
	Cameras []config.Camera
	Sources map[string]*ingest.Source
	Index   *index.Index
	Root    string // recordings root
	Log     *slog.Logger
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /api/cameras", s.cameras)
	mux.HandleFunc("GET /api/status", s.status)
	mux.HandleFunc("GET /api/recordings/{camera}", s.recordings)
	mux.HandleFunc("GET /recordings/{path...}", s.recordingFile)
	return mux
}

type cameraJSON struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

func (s *Server) cameras(w http.ResponseWriter, _ *http.Request) {
	out := make([]cameraJSON, len(s.Cameras))
	for i, c := range s.Cameras {
		out[i] = cameraJSON{ID: c.ID, Label: c.Label}
	}
	writeJSON(w, out)
}

func (s *Server) status(w http.ResponseWriter, _ *http.Request) {
	out := map[string]ingest.Status{}
	for id, src := range s.Sources {
		out[id] = src.Status()
	}
	writeJSON(w, map[string]any{"cameras": out})
}

type segmentJSON struct {
	Start int64  `json:"start"` // Unix ms
	End   int64  `json:"end"`   // Unix ms
	URL   string `json:"url"`
	Size  int64  `json:"size"`
}

func (s *Server) recordings(w http.ResponseWriter, r *http.Request) {
	camera := r.PathValue("camera")
	if _, ok := s.Sources[camera]; !ok {
		http.Error(w, "unknown camera", http.StatusNotFound)
		return
	}
	now := time.Now()
	from, err := parseTime(r.URL.Query().Get("from"), now.Add(-defaultRange))
	if err != nil {
		http.Error(w, "from: "+err.Error(), http.StatusBadRequest)
		return
	}
	to, err := parseTime(r.URL.Query().Get("to"), now)
	if err != nil {
		http.Error(w, "to: "+err.Error(), http.StatusBadRequest)
		return
	}
	entries := s.Index.Query(camera, from, to)
	out := make([]segmentJSON, len(entries))
	for i, e := range entries {
		out[i] = segmentJSON{
			Start: e.Start.UnixMilli(),
			End:   e.End().UnixMilli(),
			URL:   "/recordings/" + e.Path,
			Size:  e.Size,
		}
	}
	writeJSON(w, out)
}

// parseTime accepts Unix milliseconds or RFC 3339.
func parseTime(v string, def time.Time) (time.Time, error) {
	if v == "" {
		return def, nil
	}
	if ms, err := strconv.ParseInt(v, 10, 64); err == nil {
		return time.UnixMilli(ms), nil
	}
	return time.Parse(time.RFC3339Nano, v)
}

func (s *Server) recordingFile(w http.ResponseWriter, r *http.Request) {
	rel := r.PathValue("path")
	// Only well-formed segment paths are served; this also rules out traversal.
	if _, err := segment.Parse(rel); err != nil {
		http.NotFound(w, r)
		return
	}
	f, err := os.Open(filepath.Join(s.Root, filepath.FromSlash(rel)))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			s.Index.Remove(rel)
		}
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "video/mp4")
	// Finished segments never change.
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeContent(w, r, "", st.ModTime(), f)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
