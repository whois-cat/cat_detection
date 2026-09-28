// Package api serves streamhub's HTTP API.
package api

import (
	"cmp"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/coder/websocket"

	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/index"
	"github.com/whois-cat/cat_detection/streamhub/internal/ingest"
	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/live"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/sidecar"
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
	Bus     *labels.Bus
	Summary *sidecar.Summary
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
	mux.HandleFunc("GET /api/detections/{camera}", s.detections)
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
	// Labels is the segment's CV result sidecar (404 if none was written).
	Labels string `json:"labels"`
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
			Start:  e.Start.UnixMilli(),
			End:    e.End().UnixMilli(),
			URL:    "/recordings/" + e.Path,
			Size:   e.Size,
			Labels: "/recordings/" + e.Sidecar(),
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

// detections returns detection counts per cat, aggregated into buckets of
// `bucket` ms (default and minimum 1000): [[bucketStartMs, cat, count], ...]
// sorted by time, then cat.
func (s *Server) detections(w http.ResponseWriter, r *http.Request) {
	camera, from, to, ok := s.cameraAndRange(w, r)
	if !ok {
		return
	}
	bucket := int64(1000)
	if v := r.URL.Query().Get("bucket"); v != "" {
		ms, err := strconv.ParseInt(v, 10, 64)
		if err != nil || ms <= 0 {
			http.Error(w, "bucket: want milliseconds", http.StatusBadRequest)
			return
		}
		bucket = max(bucket, ms)
	}
	type key struct {
		start int64
		cat   string
	}
	counts := map[key]int{}
	var order []key
	for _, b := range s.Summary.Query(camera, from, to) {
		k := key{b.Sec * 1000 / bucket * bucket, b.Cat}
		if _, seen := counts[k]; !seen {
			order = append(order, k)
		}
		counts[k] += b.N
	}
	// Query is sorted by (second, cat); keep time order across merged buckets.
	slices.SortStableFunc(order, func(a, b key) int { return cmp.Or(cmp.Compare(a.start, b.start), strings.Compare(a.cat, b.cat)) })
	out := make([][3]any, len(order))
	for i, k := range order {
		out[i] = [3]any{k.start, k.cat, counts[k]}
	}
	writeJSON(w, out)
}

func (s *Server) live(w http.ResponseWriter, r *http.Request) {
	camera := r.PathValue("camera")
	stream, ok := s.Streams[camera]
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
	if err := live.Serve(ctx, conn, camera, stream, s.Bus, s.Log.With("camera", camera)); err != nil {
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
	// Only well-formed segment and sidecar paths are served; this also rules
	// out traversal.
	_, segErr := segment.Parse(rel)
	_, _, sidecarErr := segment.ParseSidecar(rel)
	if segErr != nil && sidecarErr != nil {
		http.NotFound(w, r)
		return
	}
	f, err := os.Open(filepath.Join(s.Root, filepath.FromSlash(rel)))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) && segErr == nil {
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
	if segErr == nil {
		w.Header().Set("Content-Type", "video/mp4")
		// Finished segments never change.
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		// Sidecars grow while results arrive.
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")
	}
	http.ServeContent(w, r, "", st.ModTime(), f)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
