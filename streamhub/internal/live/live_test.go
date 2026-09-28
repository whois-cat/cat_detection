package live

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/coder/websocket"

	"github.com/whois-cat/cat_detection/streamhub/internal/media"
)

func loadAUs(t *testing.T) [][][]byte {
	t.Helper()
	data, err := os.ReadFile("../../testdata/tiny.h264")
	if err != nil {
		t.Fatal(err)
	}
	var aus [][][]byte
	for _, n := range bytes.Split(data, []byte{0, 0, 1}) {
		n = bytes.TrimRight(n, "\x00")
		if len(n) == 0 {
			continue
		}
		if h264.NALUType(n[0]&0x1f) == h264.NALUTypeAccessUnitDelimiter {
			aus = append(aus, nil)
			continue
		}
		aus[len(aus)-1] = append(aus[len(aus)-1], n)
	}
	return aus
}

func TestLiveStream(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not available")
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	aus := loadAUs(t)
	stream := media.NewStream()
	var sps, pps []byte
	var pts []int64
	publish := func(i int, newSession bool) {
		for _, n := range aus[i] {
			switch h264.NALUType(n[0] & 0x1f) {
			case h264.NALUTypeSPS:
				sps = n
			case h264.NALUTypePPS:
				pps = n
			}
		}
		p := int64(160_000_000_000_000) + int64(i)*6000
		pts = append(pts, p)
		stream.Publish(&media.Frame{PTS: p, AU: aus[i], IDR: h264.IsRandomAccess(aus[i]), SPS: sps, PPS: pps, NewSession: newSession})
	}
	// Frames 0..12 before the viewer connects: the cached GOP is 10..12.
	for i := range 13 {
		publish(i, i == 0)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		Serve(conn.CloseRead(r.Context()), conn, stream, log)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.CloseNow()
	// Wait until the cached GOP arrived, so the rest is live.
	var msgs [][]byte
	read := func() {
		_, b, err := conn.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		msgs = append(msgs, b)
	}
	read() // init
	read() // frame 10
	read() // frame 11
	// A reconnect at 25 (mid-GOP: resumes at the IDR at 30) keeps the stream playable.
	for i := 13; i < 45; i++ {
		publish(i, i == 25)
	}
	// Frames 12..24 (24 flushed by the reconnect) and 30..43 (44 is held back
	// waiting for its successor); frames 25..29 wait for the IDR.
	for range (24 - 12 + 1) + (43 - 30 + 1) {
		read()
	}
	if string(msgs[0][4:8]) != "ftyp" {
		t.Fatalf("first message is %q, want init segment", msgs[0][4:8])
	}

	file := filepath.Join(t.TempDir(), "live.mp4")
	os.WriteFile(file, bytes.Join(msgs, nil), 0o644)
	out, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "v",
		"-show_entries", "packet=pts", "-of", "csv=p=0", file).Output()
	if err != nil {
		t.Fatal(err)
	}
	var got []int64
	for _, l := range strings.Fields(string(out)) {
		v, _ := strconv.ParseInt(l, 10, 64)
		got = append(got, v)
	}
	want := slices.Concat(pts[10:25], pts[30:44])
	if !slices.Equal(got, want) {
		t.Errorf("live PTS\n got %v\nwant %v", got, want)
	}
	if e, _ := exec.Command("ffmpeg", "-v", "error", "-i", file, "-enc_time_base:v", "1/90000", "-f", "null", "-").CombinedOutput(); len(e) > 0 {
		t.Errorf("decode errors: %s", e)
	}
}
