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
	"strings"
	"time"

	"github.com/coder/websocket"

	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/index"
	"github.com/whois-cat/cat_detection/streamhub/internal/ingest"
	"github.com/whois-cat/cat_detection/streamhub/internal/live"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
)

const (
	// defaultRange is the query window when from/to are omitted.
	defaultRange = time.Hour
	// defaultMaxGap merges recorded spans separated by less than this.
	defaultMaxGap = time.Second
)

// Server holds what the handlers need.
type Server struct {
	Cameras []config.Camera
	Sources map[string]*ingest.Source
	Streams map[string]*media.Stream
	Index   *index.Index
	Root    string // recordings root
	// WebUIDir holds the built webui; empty disables serving it.
	WebUIDir string
	Log      *slog.Logger
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /api/cameras", s.cameras)
	mux.HandleFunc("GET /api/status", s.status)
	mux.HandleFunc("GET /api/recordings/{camera}", s.recordings)
	mux.HandleFunc("GET /api/ranges/{camera}", s.ranges)
	mux.HandleFunc("GET /api/live/{camera}", s.live)
	mux.HandleFunc("GET /recordings/{path...}", s.recordingFile)
	if s.WebUIDir != "" {
		mux.Handle("GET /", spa(s.WebUIDir))
	}
	return mux
}

type cameraJSON struct {
	ID string `json:"id"`
}

func (s *Server) cameras(w http.ResponseWriter, _ *http.Request) {
	out := make([]cameraJSON, len(s.Cameras))
	for i, c := range s.Cameras {
		out[i] = cameraJSON{ID: c.ID}
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

// cameraAndRange parses the {camera} path value and from/to query parameters,
// writing an error response and returning ok=false if they're invalid.
func (s *Server) cameraAndRange(w http.ResponseWriter, r *http.Request) (camera string, from, to time.Time, ok bool) {
	camera = r.PathValue("camera")
	if _, known := s.Sources[camera]; !known {
		http.Error(w, "unknown camera", http.StatusNotFound)
		return
	}
	now := time.Now()
	from, err := parseTime(r.URL.Query().Get("from"), now.Add(-defaultRange))
	if err != nil {
		http.Error(w, "from: "+err.Error(), http.StatusBadRequest)
		return
	}
	to, err = parseTime(r.URL.Query().Get("to"), now)
	if err != nil {
		http.Error(w, "to: "+err.Error(), http.StatusBadRequest)
		return
	}
	return camera, from, to, true
}

func (s *Server) recordings(w http.ResponseWriter, r *http.Request) {
	camera, from, to, ok := s.cameraAndRange(w, r)
	if !ok {
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

func (s *Server) ranges(w http.ResponseWriter, r *http.Request) {
	camera, from, to, ok := s.cameraAndRange(w, r)
	if !ok {
		return
	}
	maxGap := defaultMaxGap
	if v := r.URL.Query().Get("gap"); v != "" {
		ms, err := strconv.ParseInt(v, 10, 64)
		if err != nil || ms < 0 {
			http.Error(w, "gap: want milliseconds", http.StatusBadRequest)
			return
		}
		maxGap = time.Duration(ms) * time.Millisecond
	}
	rs := s.Index.Ranges(camera, from, to, maxGap)
	out := make([][2]int64, len(rs))
	for i, rg := range rs {
		out[i] = [2]int64{rg.Start.UnixMilli(), rg.End.UnixMilli()}
	}
	writeJSON(w, out)
}

func (s *Server) live(w http.ResponseWriter, r *http.Request) {
	stream, ok := s.Streams[r.PathValue("camera")]
	if !ok {
		http.Error(w, "unknown camera", http.StatusNotFound)
		return
	}
	conn, err := websocket.Accept(w, r, nil)
	if err != nil {
		return // Accept has written the response
	}
	defer conn.CloseNow()
	// Nothing is read from the client; CloseRead handles control frames and
	// cancels ctx when the client goes away.
	ctx := conn.CloseRead(r.Context())
	if err := live.Serve(ctx, conn, stream, s.Log.With("camera", r.PathValue("camera"))); err != nil {
		s.Log.Debug("live viewer disconnected", "err", err)
	}
	conn.Close(websocket.StatusNormalClosure, "")
}

// spa serves a single-page app: existing files as-is (hashed assets are
// immutable), anything else falls back to index.html.
func spa(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := filepath.Join(dir, filepath.FromSlash(filepath.Clean("/"+r.URL.Path)))
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			w.Header().Set("Cache-Control", "no-cache")
			http.ServeFile(w, r, filepath.Join(dir, "index.html"))
			return
		}
		if strings.HasPrefix(r.URL.Path, "/assets/") {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		}
		files.ServeHTTP(w, r)
	})
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
