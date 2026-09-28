package hub

import (
	"bufio"
	"context"
	"encoding/binary"
	"io"
	"log/slog"
	"net"
	"testing"
	"time"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
)

type testClient struct {
	t    *testing.T
	conn net.Conn
	r    *bufio.Reader
}

func (c *testClient) send(v any) {
	c.t.Helper()
	b, err := msgpack.Marshal(v)
	if err != nil {
		c.t.Fatal(err)
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := c.conn.Write(append(hdr[:], b...)); err != nil {
		c.t.Fatal(err)
	}
}

func (c *testClient) recv() map[string]any {
	c.t.Helper()
	c.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	var hdr [4]byte
	if _, err := io.ReadFull(c.r, hdr[:]); err != nil {
		c.t.Fatal(err)
	}
	body := make([]byte, binary.BigEndian.Uint32(hdr[:]))
	if _, err := io.ReadFull(c.r, body); err != nil {
		c.t.Fatal(err)
	}
	var m map[string]any
	if err := msgpack.Unmarshal(body, &m); err != nil {
		c.t.Fatal(err)
	}
	return m
}

func TestCVWorkerSession(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	stream := media.NewStream()
	bus := labels.NewBus()
	s := &Server{
		Streams: map[string]*media.Stream{"grey": stream, "beige": media.NewStream()},
		Config:  map[string]map[string]any{"grey": {"rotate_deg": 90}},
		Bus:     bus,
		Log:     log,
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Serve(ctx, ln)

	sps := []byte{0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x41, 0x0d, 0xa1, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x03, 0x00, 0x1e, 0x8f, 0x12, 0x25, 0x96}
	pps := []byte{0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0}
	// A cached GOP (IDR + P) exists before the worker connects.
	stream.Publish(&media.Frame{PTS: 100, IDR: true, AU: [][]byte{{0x65, 1}}, SPS: sps, PPS: pps, NewSession: true})
	stream.Publish(&media.Frame{PTS: 200, AU: [][]byte{{0x41, 2}}, SPS: sps, PPS: pps})

	results := bus.Subscribe(10)
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	c := &testClient{t: t, conn: conn, r: bufio.NewReader(conn)}
	c.send(map[string]any{"type": "hello", "role": "cv", "id": "w1", "cameras": []string{"grey"},
		"model": map[string]any{"name": "yolo_cat", "version": "20260630"}})

	m := c.recv()
	if m["type"] != "stream" || m["camera"] != "grey" || m["config"].(map[string]any)["rotate_deg"] == nil {
		t.Fatalf("want stream message with config, got %v", m)
	}
	for _, want := range []int64{100, 200} {
		m := c.recv()
		if m["type"] != "frame" || toInt(m["pts"]) != want {
			t.Fatalf("want frame %d, got %v", want, m)
		}
		if data := m["data"].([]byte); data[3] != 1 || data[0] != 0 {
			t.Fatalf("frame data not Annex-B: %x", data)
		}
	}
	if !bus.CVActive("grey") || bus.CVActive("beige") {
		t.Fatal("attachment not tracked")
	}

	c.send(map[string]any{"type": "result", "camera": "grey", "pts": 200, "infer_ms": 12.5,
		"dets": []map[string]any{{"box": []float64{0.1, 0.2, 0.3, 0.4}, "score": 0.9, "cats": map[string]float64{"alisa": 0.8, "chuzh": 0.2}}}})
	c.send(map[string]any{"type": "result", "camera": "beige", "pts": 1}) // not served: ignored
	select {
	case r := <-results:
		if r.Camera != "grey" || r.PTS != 200 || r.Worker != "w1" || r.Model != "yolo_cat@20260630" ||
			len(r.Dets) != 1 || r.Dets[0].Box[3] != 0.4 || r.Dets[0].TopCat() != "alisa" {
			t.Fatalf("result = %+v", r)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no result on the bus")
	}
	select {
	case r := <-results:
		t.Fatalf("unexpected result %+v", r)
	case <-time.After(100 * time.Millisecond):
	}

	conn.Close()
	deadline := time.Now().Add(5 * time.Second)
	for bus.CVActive("grey") && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if bus.CVActive("grey") {
		t.Fatal("worker still attached after disconnect")
	}
}

func TestRejects(t *testing.T) {
	s := &Server{Streams: map[string]*media.Stream{"grey": media.NewStream()}, Bus: labels.NewBus(),
		Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Serve(ctx, ln)
	for _, hello := range []map[string]any{
		{"type": "result"},
		{"type": "hello", "role": "nope"},
		{"type": "hello", "role": "cv", "cameras": []string{"pink"}},
	} {
		conn, _ := net.Dial("tcp", ln.Addr().String())
		c := &testClient{t: t, conn: conn, r: bufio.NewReader(conn)}
		c.send(hello)
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		if _, err := c.r.ReadByte(); err != io.EOF {
			t.Errorf("hello %v: connection not closed (err %v)", hello, err)
		}
		conn.Close()
	}
}

func toInt(v any) int64 {
	switch x := v.(type) {
	case int8:
		return int64(x)
	case int16:
		return int64(x)
	case int32:
		return int64(x)
	case int64:
		return x
	case uint8:
		return int64(x)
	case uint16:
		return int64(x)
	case uint32:
		return int64(x)
	case uint64:
		return int64(x)
	}
	return -1
}
