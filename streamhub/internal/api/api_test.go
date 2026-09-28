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
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
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
		Cameras: []config.Camera{{ID: "grey", Label: "Grey"}},
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
