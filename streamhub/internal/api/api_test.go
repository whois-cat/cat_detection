package api

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/index"
	"github.com/whois-cat/cat_detection/streamhub/internal/ingest"
	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
	"github.com/whois-cat/cat_detection/streamhub/internal/sidecar"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

func TestRecordingsAndFiles(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	root := t.TempDir()
	start := time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)
	rel := segment.FinalPath("grey", start, 10*time.Second)
	os.MkdirAll(filepath.Join(root, filepath.Dir(rel)), 0o755)
	os.WriteFile(filepath.Join(root, rel), []byte("0123456789"), 0o644)
	idx := index.New(root, log)
	if err := idx.Scan(); err != nil {
		t.Fatal(err)
	}
	s := &Server{
		Cameras: []config.Camera{{ID: "grey"}},
		Sources: map[string]*ingest.Source{"grey": ingest.NewSource("grey", "rtsp://x", media.NewStream(), log)},
		Index:   idx,
		Root:    root,
		Log:     log,
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	get := func(path string, hdr ...string) *http.Response {
		t.Helper()
		req, _ := http.NewRequest("GET", srv.URL+path, nil)
		for i := 0; i+1 < len(hdr); i += 2 {
			req.Header.Set(hdr[i], hdr[i+1])
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	from := strconv.FormatInt(start.Add(-time.Minute).UnixMilli(), 10)
	resp := get("/api/recordings/grey?from=" + from + "&to=" + start.Add(time.Minute).Format(time.RFC3339))
	var segs []segmentJSON
	json.NewDecoder(resp.Body).Decode(&segs)
	resp.Body.Close()
	if len(segs) != 1 || segs[0].Start != start.UnixMilli() || segs[0].End != start.Add(10*time.Second).UnixMilli() {
		t.Fatalf("recordings = %+v", segs)
	}

	// Range request on the segment file.
	resp = get(segs[0].URL, "Range", "bytes=2-4")
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent || string(body) != "234" {
		t.Fatalf("range: %d %q", resp.StatusCode, body)
	}

	for _, p := range []string{"/recordings/../../etc/passwd", "/recordings/grey/", "/api/recordings/nope"} {
		if resp := get(p); resp.StatusCode != http.StatusNotFound {
			t.Errorf("%s: status %d", p, resp.StatusCode)
		}
	}

	// A file deleted behind our back: 404 and dropped from the index.
	os.Remove(filepath.Join(root, rel))
	if resp := get(segs[0].URL); resp.StatusCode != http.StatusNotFound {
		t.Errorf("deleted file: status %d", resp.StatusCode)
	}
	if got := idx.Query("grey", start, start.Add(time.Minute)); len(got) != 0 {
		t.Errorf("deleted file still indexed")
	}
}

func TestDetectionsBuckets(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	sum := sidecar.NewSummary(t.TempDir(), log)
	t0 := time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)
	rel := filepath.ToSlash(segment.SidecarPath("grey", t0))
	add := func(d time.Duration, cats ...string) {
		r := labels.Result{Camera: "grey", PTS: timeline.TimeToTicks(t0.Add(d))}
		for _, c := range cats {
			r.Dets = append(r.Dets, labels.Det{Cats: map[string]float64{c: 1}})
		}
		sum.Add(rel, r)
	}
	add(0, "b", "a")
	add(1500*time.Millisecond, "a")
	add(12*time.Second, "a")
	s := &Server{Sources: map[string]*ingest.Source{"grey": nil}, Summary: sum, Log: log}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	get := func(q string) [][3]any {
		t.Helper()
		resp, err := http.Get(srv.URL + "/api/detections/grey?from=" + strconv.FormatInt(t0.UnixMilli(), 10) +
			"&to=" + strconv.FormatInt(t0.Add(time.Minute).UnixMilli(), 10) + q)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out [][3]any
		json.NewDecoder(resp.Body).Decode(&out)
		return out
	}
	ms := float64(t0.UnixMilli())
	if got := get(""); len(got) != 4 || got[0][1] != "a" || got[1][1] != "b" || got[2][0] != ms+1000 || got[3][0] != ms+12000 {
		t.Errorf("per-second: %v", got)
	}
	got := get("&bucket=10000")
	if len(got) != 3 || got[0][2] != 2.0 || got[0][1] != "a" || got[1][1] != "b" || got[2][0] != ms+10000 {
		t.Errorf("10 s buckets: %v", got)
	}
	if got := get("&bucket=10"); len(got) != 4 { // below 1 s is clamped to 1 s
		t.Errorf("clamped bucket: %v", got)
	}
}
