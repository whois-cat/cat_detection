// Package ingest pulls H.264 video from a camera over RTSP and publishes
// frames with wall-clock PTS.
package ingest

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/gortsplib/v5/pkg/base"
	"github.com/bluenviron/gortsplib/v5/pkg/format"
	"github.com/bluenviron/gortsplib/v5/pkg/format/rtph264"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/pion/rtp"

	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

const (
	minBackoff = time.Second
	maxBackoff = 30 * time.Second
	// A session that lasted this long resets the backoff.
	stableSession = 30 * time.Second
	readTimeout   = 10 * time.Second
	// fpsSmoothing is the EWMA weight of the newest frame interval.
	fpsSmoothing = 0.05
)

// Status is a snapshot of a source's state.
type Status struct {
	Connected      bool          `json:"connected"`
	ConnectedSince time.Time     `json:"connected_since,omitzero"`
	Sessions       int           `json:"sessions"`
	Frames         uint64        `json:"frames"`
	FPS            float64       `json:"fps"`
	Width          int           `json:"width,omitempty"`
	Height         int           `json:"height,omitempty"`
	LastFrame      time.Time     `json:"last_frame,omitzero"`
	SlewBacklog    time.Duration `json:"slew_backlog_ns"`
	LastError      string        `json:"last_error,omitempty"`
}

// Source ingests one camera.
type Source struct {
	camera string
	url    string
	out    *media.Stream
	log    *slog.Logger

	mu     sync.Mutex
	status Status
}

// NewSource returns a Source publishing the camera's frames to out.
func NewSource(camera, url string, out *media.Stream, log *slog.Logger) *Source {
	return &Source{camera: camera, url: url, out: out, log: log.With("camera", camera)}
}

// Status returns a snapshot of the source's state.
func (s *Source) Status() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.status
}

// Run ingests until ctx is cancelled, reconnecting with backoff.
func (s *Source) Run(ctx context.Context) {
	tl := timeline.New(timeline.DefaultConfig())
	backoff := minBackoff
	for ctx.Err() == nil {
		started := time.Now()
		err := s.session(ctx, tl)
		s.mu.Lock()
		s.status.Connected = false
		if err != nil && ctx.Err() == nil {
			s.status.LastError = err.Error()
		}
		s.mu.Unlock()
		if ctx.Err() != nil {
			return
		}
		if time.Since(started) > stableSession {
			backoff = minBackoff
		}
		s.log.Warn("camera session ended, reconnecting", "err", err, "in", backoff)
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, maxBackoff)
	}
}

func (s *Source) session(ctx context.Context, tl *timeline.Timeline) error {
	u, err := base.ParseURL(s.url)
	if err != nil {
		return fmt.Errorf("parse url: %w", err)
	}
	proto := gortsplib.ProtocolTCP
	c := gortsplib.Client{
		Scheme:      u.Scheme,
		Host:        u.Host,
		Protocol:    &proto,
		ReadTimeout: readTimeout,
	}
	if err := c.Start(); err != nil {
		return err
	}
	defer c.Close()

	desc, _, err := c.Describe(u)
	if err != nil {
		return fmt.Errorf("describe: %w", err)
	}
	var forma *format.H264
	medi := desc.FindFormat(&forma)
	if medi == nil {
		return errors.New("camera offers no H.264 video")
	}
	dec, err := forma.CreateDecoder()
	if err != nil {
		return err
	}
	// Only the video media is set up: audio never leaves the camera.
	if _, err := c.Setup(desc.BaseURL, medi, 0, 0); err != nil {
		return fmt.Errorf("setup: %w", err)
	}

	tl.NewSession()
	sps, pps := forma.SPS, forma.PPS
	newSession := true
	waitingIDR := true
	var lastPTS int64

	c.OnPacketRTP(medi, forma, func(pkt *rtp.Packet) {
		au, err := dec.Decode(pkt)
		if err != nil {
			if !errors.Is(err, rtph264.ErrMorePacketsNeeded) && !errors.Is(err, rtph264.ErrNonStartingPacketAndNoPrevious) {
				s.log.Debug("rtp decode", "err", err)
			}
			return
		}
		arrival := time.Now()
		for _, nalu := range au {
			switch h264.NALUType(nalu[0] & 0x1f) {
			case h264.NALUTypeSPS:
				sps = nalu
			case h264.NALUTypePPS:
				pps = nalu
			}
		}
		// Every frame goes through the timeline so its estimate sees all arrivals.
		pts := tl.Next(pkt.Timestamp, arrival)
		idr := h264.IsRandomAccess(au)
		if waitingIDR && (!idr || sps == nil || pps == nil) {
			return
		}
		waitingIDR = false
		s.out.Publish(&media.Frame{
			Camera: s.camera, PTS: pts, AU: au, IDR: idr,
			SPS: sps, PPS: pps, NewSession: newSession,
		})
		s.updateStatus(newSession, pts, lastPTS, sps, tl.Stats())
		newSession = false
		lastPTS = pts
	})

	if _, err := c.Play(nil); err != nil {
		return fmt.Errorf("play: %w", err)
	}
	s.log.Info("camera connected")

	errc := make(chan error, 1)
	go func() { errc <- c.Wait() }()
	select {
	case <-ctx.Done():
		c.Close()
		<-errc
		return nil
	case err := <-errc:
		return err
	}
}

func (s *Source) updateStatus(newSession bool, pts, lastPTS int64, sps []byte, ts timeline.Stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := &s.status
	if newSession {
		st.Connected = true
		st.ConnectedSince = time.Now()
		st.Sessions++
		st.LastError = ""
		var p h264.SPS
		if err := p.Unmarshal(sps); err == nil {
			st.Width, st.Height = p.Width(), p.Height()
		}
	} else if dt := timeline.TicksToDuration(pts - lastPTS).Seconds(); dt > 0 {
		if st.FPS == 0 {
			st.FPS = 1 / dt
		} else {
			st.FPS = 1 / ((1-fpsSmoothing)/st.FPS + fpsSmoothing*dt)
		}
	}
	st.Frames++
	st.LastFrame = timeline.TicksToTime(pts)
	st.SlewBacklog = ts.SlewBacklog
}
