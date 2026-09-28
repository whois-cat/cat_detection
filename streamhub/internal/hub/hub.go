// Package hub serves the internal protocol for non-browser clients: CV
// workers and the decider.
//
// Transport: TCP; each message is a 4-byte big-endian length followed by a
// msgpack map with a "type" key. Clients connect and send "hello" first:
//
//	C→H hello    {role: "cv" | "decider", id, cameras: ["*"] | [ids],
//	             model: {name, version} (cv only)}
//	both ping    {} — liveness, every pingEvery
//
// role "cv":
//
//	H→C stream   {camera, sps, pps, width, height, config} — before the first
//	             frame of a camera and whenever its parameter sets change
//	H→C frame    {camera, pts, key, data} — every frame (Annex-B), starting at
//	             a keyframe; after dropped frames it resumes at the next keyframe
//	C→H result   {camera, pts, infer_ms, dets: [{box, score, cats, emb?}]}
//
// role "decider":
//
//	H→C result   {camera, pts, model, worker, infer_ms, dets} — CV results of
//	             its cameras as they arrive
//	C→H decision {camera, pts, …} — stored next to the video and shown live
package hub

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"slices"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/whois-cat/cat_detection/streamhub/internal/labels"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
)

const (
	maxMessage   = 16 << 20
	helloTimeout = 10 * time.Second
	pingEvery    = 10 * time.Second
	// A client silent for this long (no results, no pings) is dropped.
	readTimeout  = 3 * pingEvery
	writeTimeout = 30 * time.Second
	// Frames a client may lag per camera before frames are dropped.
	frameBuffer = 120
	outQueue    = 64
)

// Server serves the hub protocol.
type Server struct {
	Streams map[string]*media.Stream
	// Config is passed to CV workers per camera (rotation, detection area, …).
	Config map[string]map[string]any
	Ch     *labels.Channels
	Log    *slog.Logger
}

type envelope struct {
	Type string `msgpack:"type"`
}

type helloMsg struct {
	Role    string   `msgpack:"role"`
	ID      string   `msgpack:"id"`
	Cameras []string `msgpack:"cameras"`
	Model   struct {
		Name    string `msgpack:"name"`
		Version string `msgpack:"version"`
	} `msgpack:"model"`
}

type streamMsg struct {
	Type   string         `msgpack:"type"`
	Camera string         `msgpack:"camera"`
	SPS    []byte         `msgpack:"sps"`
	PPS    []byte         `msgpack:"pps"`
	Width  int            `msgpack:"width"`
	Height int            `msgpack:"height"`
	Config map[string]any `msgpack:"config"`
}

type resultMsg struct {
	Type    string       `msgpack:"type"`
	Camera  string       `msgpack:"camera"`
	PTS     int64        `msgpack:"pts"`
	Model   string       `msgpack:"model"`
	Worker  string       `msgpack:"worker"`
	InferMs float64      `msgpack:"infer_ms"`
	Dets    []labels.Det `msgpack:"dets"`
}

type frameMsg struct {
	Type   string `msgpack:"type"`
	Camera string `msgpack:"camera"`
	PTS    int64  `msgpack:"pts"`
	Key    bool   `msgpack:"key"`
	Data   []byte `msgpack:"data"`
}

// Serve accepts clients until ctx is done.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	go func() { <-ctx.Done(); ln.Close() }()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go s.handle(ctx, conn)
	}
}

type client struct {
	conn net.Conn
	r    *bufio.Reader
	out  chan []byte
	log  *slog.Logger
}

func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	c := &client{conn: conn, r: bufio.NewReader(conn), out: make(chan []byte, outQueue),
		log: s.Log.With("client", conn.RemoteAddr().String())}

	conn.SetReadDeadline(time.Now().Add(helloTimeout))
	var hello helloMsg
	typ, body, err := c.read()
	if err == nil && typ != "hello" {
		err = fmt.Errorf("first message is %q, want hello", typ)
	}
	if err == nil {
		err = msgpack.Unmarshal(body, &hello)
	}
	if err == nil && hello.Role != "cv" && hello.Role != "decider" {
		err = fmt.Errorf("unsupported role %q", hello.Role)
	}
	cameras, cerr := s.resolveCameras(hello.Cameras)
	if err = errors.Join(err, cerr); err != nil {
		c.log.Warn("rejecting hub client", "err", err)
		return
	}
	model := hello.Model.Name
	if hello.Model.Version != "" {
		model += "@" + hello.Model.Version
	}
	c.log = c.log.With("role", hello.Role, "id", hello.ID)
	if hello.Role == "cv" {
		c.log = c.log.With("model", model)
	}
	c.log.Info("hub client connected", "cameras", cameras)
	defer c.log.Info("hub client disconnected")

	go c.writeLoop(ctx, cancel)
	go c.pingLoop(ctx)
	if hello.Role == "cv" {
		for _, cam := range cameras {
			detach := s.Ch.CV.Attach(cam)
			defer detach()
			go s.forward(ctx, c, cam)
		}
	} else {
		go s.forwardResults(ctx, c, cameras)
	}

	for {
		conn.SetReadDeadline(time.Now().Add(readTimeout))
		typ, body, err := c.read()
		if err != nil {
			if ctx.Err() == nil && !errors.Is(err, io.EOF) {
				c.log.Warn("hub client read failed", "err", err)
			}
			return
		}
		switch {
		case typ == "result" && hello.Role == "cv":
			var r labels.Result
			if err := msgpack.Unmarshal(body, &r); err != nil {
				c.log.Warn("bad result", "err", err)
				return
			}
			if !slices.Contains(cameras, r.Camera) {
				c.log.Warn("result for a camera this worker doesn't serve", "camera", r.Camera)
				continue
			}
			r.Model, r.Worker = model, hello.ID
			s.Ch.Results.Publish(r)
		case typ == "decision" && hello.Role == "decider":
			d, err := decodeDecision(body)
			if err != nil {
				c.log.Warn("bad decision", "err", err)
				continue
			}
			if !slices.Contains(cameras, d.Camera) {
				c.log.Warn("decision for a camera this decider doesn't serve", "camera", d.Camera)
				continue
			}
			d.Fields["decider"] = hello.ID
			s.Ch.PublishDecision(d)
		case typ == "ping":
		default:
			c.log.Debug("ignoring message", "type", typ)
		}
	}
}

// forwardResults sends the CV results of cameras to a decider.
func (s *Server) forwardResults(ctx context.Context, c *client, cameras []string) {
	results := s.Ch.Results.Subscribe(1024)
	defer s.Ch.Results.Unsubscribe(results)
	for {
		select {
		case <-ctx.Done():
			return
		case r := <-results:
			if !slices.Contains(cameras, r.Camera) {
				continue
			}
			if !c.send(ctx, resultMsg{Type: "result", Camera: r.Camera, PTS: r.PTS, Model: r.Model,
				Worker: r.Worker, InferMs: r.InferMs, Dets: r.Dets}) {
				return
			}
		}
	}
}

func decodeDecision(body []byte) (labels.Decision, error) {
	var m map[string]any
	if err := msgpack.Unmarshal(body, &m); err != nil {
		return labels.Decision{}, err
	}
	camera, _ := m["camera"].(string)
	pts, ok := toInt64(m["pts"])
	if camera == "" || !ok {
		return labels.Decision{}, errors.New("decision needs camera and pts")
	}
	delete(m, "type")
	m["pts"] = pts
	return labels.Decision{Camera: camera, PTS: pts, Fields: m}, nil
}

func toInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case int8:
		return int64(x), true
	case int16:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case uint8:
		return int64(x), true
	case uint16:
		return int64(x), true
	case uint32:
		return int64(x), true
	case uint64:
		return int64(x), true
	}
	return 0, false
}

func (s *Server) resolveCameras(req []string) ([]string, error) {
	if len(req) == 0 || slices.Contains(req, "*") {
		all := make([]string, 0, len(s.Streams))
		for id := range s.Streams {
			all = append(all, id)
		}
		slices.Sort(all)
		return all, nil
	}
	for _, id := range req {
		if _, ok := s.Streams[id]; !ok {
			return nil, fmt.Errorf("unknown camera %q", id)
		}
	}
	return req, nil
}

// forward sends camera's frames to the client.
func (s *Server) forward(ctx context.Context, c *client, camera string) {
	sub, gop := s.Streams[camera].SubscribeFromGOP(frameBuffer)
	defer s.Streams[camera].Unsubscribe(sub)
	var sps, pps []byte
	waitingIDR := true
	send := func(f *media.Frame) bool {
		if waitingIDR {
			if !f.IDR {
				return true
			}
			waitingIDR = false
		}
		if !bytes.Equal(f.SPS, sps) || !bytes.Equal(f.PPS, pps) {
			sps, pps = f.SPS, f.PPS
			m := streamMsg{Type: "stream", Camera: camera, SPS: sps, PPS: pps, Config: s.Config[camera]}
			var p h264.SPS
			if err := p.Unmarshal(sps); err == nil {
				m.Width, m.Height = p.Width(), p.Height()
			}
			if !c.send(ctx, m) {
				return false
			}
		}
		return c.send(ctx, frameMsg{Type: "frame", Camera: camera, PTS: f.PTS, Key: f.IDR, Data: annexB(f.AU)})
	}
	for _, f := range gop {
		if !send(f) {
			return
		}
	}
	for {
		select {
		case <-ctx.Done():
			return
		case d, ok := <-sub.C:
			if !ok {
				return
			}
			if d.Gap {
				waitingIDR = true
			}
			if !send(d.Frame) {
				return
			}
		}
	}
}

func annexB(au [][]byte) []byte {
	var b bytes.Buffer
	for _, n := range au {
		b.Write([]byte{0, 0, 0, 1})
		b.Write(n)
	}
	return b.Bytes()
}

// send queues a message; it blocks while the client is slow (the stream then
// drops frames for it) and returns false once the connection is done.
func (c *client) send(ctx context.Context, v any) bool {
	b, err := msgpack.Marshal(v)
	if err != nil {
		c.log.Error("encoding hub message", "err", err)
		return false
	}
	select {
	case c.out <- b:
		return true
	case <-ctx.Done():
		return false
	}
}

func (c *client) writeLoop(ctx context.Context, cancel context.CancelFunc) {
	defer cancel()
	var hdr [4]byte
	for {
		select {
		case <-ctx.Done():
			return
		case b := <-c.out:
			binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
			c.conn.SetWriteDeadline(time.Now().Add(writeTimeout))
			if _, err := c.conn.Write(append(hdr[:], b...)); err != nil {
				return
			}
		}
	}
}

func (c *client) pingLoop(ctx context.Context) {
	t := time.NewTicker(pingEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if !c.send(ctx, envelope{Type: "ping"}) {
				return
			}
		}
	}
}

// read reads one message, returning its type and raw body.
func (c *client) read() (string, []byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(c.r, hdr[:]); err != nil {
		return "", nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > maxMessage {
		return "", nil, fmt.Errorf("message of %d bytes exceeds limit", n)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(c.r, body); err != nil {
		return "", nil, err
	}
	var env envelope
	if err := msgpack.Unmarshal(body, &env); err != nil {
		return "", nil, err
	}
	return env.Type, body, nil
}
