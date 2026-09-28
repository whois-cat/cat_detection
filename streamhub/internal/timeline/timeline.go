// Package timeline maps a camera's RTP timestamps to wall-clock presentation
// timestamps (PTS in 90 kHz ticks since the Unix epoch).
//
// Measured camera behaviour (see testdata/camera-timing): RTP timestamps are
// honest capture times, but the camera clock drifts relative to the host
// (tens to ~170 ppm), packets arrive in bursts delayed by up to ~1 s, and RTCP
// sender reports are unusable. So:
//
//   - frame spacing comes from RTP (never rewritten to a constant rate);
//   - the RTP→wall offset is estimated from arrival times using the lower
//     envelope (minimum arrival lag over a sliding window), since bursts only
//     ever delay packets;
//   - the offset in use follows that estimate by slewing (bounded change per
//     tick of media time), never by jumping backwards, so PTS is strictly
//     increasing for the lifetime of the Timeline, across reconnects.
package timeline

import (
	"math"
	"time"
)

// ClockRate is the H.264 RTP clock rate and the PTS time base.
const ClockRate = 90000

// Config tunes offset estimation.
type Config struct {
	// Window is the span of media time over which the minimum arrival lag is taken.
	Window time.Duration
	// MaxSlewPPM bounds how fast the offset may change in steady state
	// (parts per million of media time). Must exceed the worst camera drift.
	MaxSlewPPM float64
	// Warmup is how long after a session starts the faster WarmupSlewPPM applies.
	// The first frames of a session arrive as a burst (the camera flushes its
	// buffer), so the initial estimate is too late and must be corrected quickly.
	Warmup        time.Duration
	WarmupSlewPPM float64
	// StepThreshold: if the estimate is ahead of the offset in use by more than
	// this, jump forward instead of slewing (e.g. host clock stepped forward).
	// Backward differences are always slewed to keep PTS increasing.
	StepThreshold time.Duration
}

// DefaultConfig returns the settings validated against the captured camera data.
func DefaultConfig() Config {
	return Config{
		Window:        30 * time.Second,
		MaxSlewPPM:    500,
		Warmup:        30 * time.Second,
		WarmupSlewPPM: 100_000,
		StepThreshold: 2 * time.Second,
	}
}

type lagSample struct {
	ext int64 // extended RTP timestamp
	lag int64 // arrival - ext, in ticks
}

// Timeline converts RTP timestamps of consecutive frames to PTS. Not safe for
// concurrent use.
type Timeline struct {
	cfg        Config
	windowTick int64
	warmupTick int64

	// per RTSP session
	inSession    bool
	lastRTP      uint32
	ext          int64
	sessionStart int64
	envelope     []lagSample // monotonic deque: lag strictly increasing front to back

	offset  float64 // PTS = ext + offset
	hasLast bool
	lastExt int64
	lastPTS int64
}

// New returns a Timeline with the given config.
func New(cfg Config) *Timeline {
	return &Timeline{
		cfg:        cfg,
		windowTick: durationToTicks(cfg.Window),
		warmupTick: durationToTicks(cfg.Warmup),
	}
}

// NewSession must be called when a new RTSP session starts (reconnect): RTP
// timestamps restart from an arbitrary value. PTS continuity is preserved.
func (t *Timeline) NewSession() {
	t.inSession = false
	t.envelope = t.envelope[:0]
}

// Next returns the PTS for a frame with the given RTP timestamp that finished
// arriving at the given time.
func (t *Timeline) Next(rtp uint32, arrival time.Time) int64 {
	firstOfSession := !t.inSession
	if firstOfSession {
		t.inSession = true
		t.ext = int64(rtp)
		t.sessionStart = t.ext
	} else {
		t.ext += int64(int32(rtp - t.lastRTP))
	}
	t.lastRTP = rtp
	ext := t.ext

	target := t.observe(ext, TimeToTicks(arrival))

	if firstOfSession {
		// Fresh anchor; clamped below so PTS keeps increasing over a reconnect.
		t.offset = float64(target)
	} else {
		dt := ext - t.lastExt
		if dt < 0 {
			dt = 0
		}
		ppm := t.cfg.MaxSlewPPM
		if ext-t.sessionStart < t.warmupTick {
			ppm = t.cfg.WarmupSlewPPM
		}
		maxAdj := float64(dt) * ppm / 1e6
		diff := float64(target) - t.offset
		switch {
		case diff > float64(durationToTicks(t.cfg.StepThreshold)):
			t.offset = float64(target)
		case diff > maxAdj:
			t.offset += maxAdj
		case diff < -maxAdj:
			t.offset -= maxAdj
		default:
			t.offset = float64(target)
		}
	}

	pts := ext + int64(math.Round(t.offset))
	if t.hasLast && pts <= t.lastPTS {
		pts = t.lastPTS + 1
		t.offset = float64(pts - ext)
	}
	t.hasLast = true
	t.lastExt = ext
	t.lastPTS = pts
	return pts
}

// observe adds a lag sample and returns the current lower-envelope offset.
func (t *Timeline) observe(ext, arrival int64) int64 {
	s := lagSample{ext: ext, lag: arrival - ext}
	for n := len(t.envelope); n > 0 && t.envelope[n-1].lag >= s.lag; n-- {
		t.envelope = t.envelope[:n-1]
	}
	t.envelope = append(t.envelope, s)
	drop := 0
	for drop < len(t.envelope)-1 && t.envelope[drop].ext < ext-t.windowTick {
		drop++
	}
	t.envelope = t.envelope[drop:]
	return t.envelope[0].lag
}

// Stats describes the current estimation state.
type Stats struct {
	// SlewBacklog is how far the offset in use still has to move to reach the
	// current estimate (positive: PTS will move later).
	SlewBacklog time.Duration
}

// Stats returns the current estimation state.
func (t *Timeline) Stats() Stats {
	if len(t.envelope) == 0 {
		return Stats{}
	}
	return Stats{SlewBacklog: TicksToDuration(t.envelope[0].lag - int64(math.Round(t.offset)))}
}

// TimeToTicks converts a wall-clock time to 90 kHz ticks since the Unix epoch.
func TimeToTicks(tm time.Time) int64 {
	ns := tm.UnixNano()
	// Split to avoid overflowing int64 (ns * 9 would overflow).
	return ns/100_000*9 + ns%100_000*9/100_000
}

// TicksToTime converts 90 kHz ticks since the Unix epoch to wall-clock time.
func TicksToTime(ticks int64) time.Time {
	return time.Unix(ticks/ClockRate, ticks%ClockRate*1e9/ClockRate).UTC()
}

// TicksToDuration converts a tick count to a duration.
func TicksToDuration(ticks int64) time.Duration {
	return time.Duration(ticks/ClockRate)*time.Second + time.Duration(ticks%ClockRate*1e9/ClockRate)
}

func durationToTicks(d time.Duration) int64 {
	return int64(d/time.Second)*ClockRate + int64(d%time.Second)*ClockRate/int64(time.Second)
}
