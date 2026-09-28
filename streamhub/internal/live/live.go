// Package live streams a camera's video to a browser over WebSocket as fMP4,
// for Media Source Extensions.
//
// Binary messages are fMP4 pieces, in order: an init segment (sent first and
// again whenever the parameter sets change), then one fragment per frame.
// Timestamps are the frames' wall-clock PTS, the same as in recorded segments.
// Text messages are reserved for JSON metadata (labels, status).
package live

import (
	"context"
	"log/slog"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4"
	"github.com/coder/websocket"

	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/mux"
)

const (
	// buffer is how many frames a viewer may lag before frames are dropped
	// (it then resumes at the next keyframe).
	buffer       = 120
	writeTimeout = 5 * time.Second
)

// Serve streams stream to conn until ctx is done or the connection fails.
func Serve(ctx context.Context, conn *websocket.Conn, stream *media.Stream, log *slog.Logger) error {
	sub, gop := stream.SubscribeFromGOP(buffer)
	defer stream.Unsubscribe(sub)

	w := &writer{ctx: ctx, conn: conn, waitingIDR: true}
	for _, f := range gop {
		if err := w.push(f, false); err != nil {
			return err
		}
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case d, ok := <-sub.C:
			if !ok {
				return nil
			}
			if d.Gap {
				log.Debug("live viewer fell behind, frames dropped")
			}
			if err := w.push(d.Frame, d.Gap); err != nil {
				return err
			}
		}
	}
}

type writer struct {
	ctx        context.Context
	conn       *websocket.Conn
	prev       *media.Frame
	lastDur    int64
	sps, pps   []byte
	seq        uint32
	waitingIDR bool
}

// push holds f back until the next frame gives its duration.
func (w *writer) push(f *media.Frame, gap bool) error {
	if gap || f.NewSession {
		if w.prev != nil && !gap {
			if err := w.emit(w.prev, w.lastDur); err != nil {
				return err
			}
		}
		w.prev = nil
		w.waitingIDR = true
	}
	if w.prev != nil {
		if err := w.emit(w.prev, f.PTS-w.prev.PTS); err != nil {
			return err
		}
	}
	w.prev = f
	return nil
}

func (w *writer) emit(f *media.Frame, dur int64) error {
	if w.waitingIDR {
		if !f.IDR {
			return nil
		}
		w.waitingIDR = false
	}
	if string(f.SPS) != string(w.sps) || string(f.PPS) != string(w.pps) {
		init, err := mux.Init(f.SPS, f.PPS)
		if err != nil {
			return err
		}
		if err := w.write(init); err != nil {
			return err
		}
		w.sps, w.pps = f.SPS, f.PPS
	}
	sample, err := mux.Sample(f.AU, f.IDR, dur)
	if err != nil {
		return err
	}
	frag, err := mux.Fragment(w.seq, f.PTS, []*fmp4.Sample{sample})
	if err != nil {
		return err
	}
	w.seq++
	w.lastDur = dur
	return w.write(frag)
}

func (w *writer) write(b []byte) error {
	ctx, cancel := context.WithTimeout(w.ctx, writeTimeout)
	defer cancel()
	return w.conn.Write(ctx, websocket.MessageBinary, b)
}
