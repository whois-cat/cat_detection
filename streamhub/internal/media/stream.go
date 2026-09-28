// Package media holds the frame type passed between streamhub components and
// a per-camera fan-out.
package media

import (
	"slices"
	"sync"
)

// Frame is one H.264 access unit with its wall-clock PTS.
type Frame struct {
	Camera string
	// PTS in 90 kHz ticks since the Unix epoch; strictly increasing per camera.
	PTS int64
	AU  [][]byte // NAL units without start codes
	IDR bool
	// SPS and PPS in effect for this frame.
	SPS, PPS []byte
	// NewSession marks the first frame after (re)connecting to the camera.
	NewSession bool
}

// Delivery is a frame as received by a subscriber.
type Delivery struct {
	*Frame
	// Gap is set when frames were dropped before this one because the
	// subscriber fell behind; the subscriber must resync at the next IDR.
	Gap bool
}

// Subscription receives frames from a Stream.
type Subscription struct {
	C       chan Delivery
	pending bool // a frame was dropped; mark the next delivered one
}

// maxGOP bounds the GOP cache (frames); longer GOPs are not cached.
const maxGOP = 600

// Stream fans frames of one camera out to subscribers. Publishing never blocks:
// a subscriber whose buffer is full misses frames and is told via Delivery.Gap.
//
// It also caches the frames since the last IDR, so a new subscriber can start
// decoding immediately instead of waiting for the next keyframe.
type Stream struct {
	mu   sync.Mutex
	subs map[*Subscription]struct{}
	gop  []*Frame
}

// NewStream returns an empty Stream.
func NewStream() *Stream {
	return &Stream{subs: map[*Subscription]struct{}{}}
}

// Subscribe registers a subscriber with the given buffer size.
func (s *Stream) Subscribe(buffer int) *Subscription {
	sub, _ := s.SubscribeFromGOP(buffer)
	return sub
}

// SubscribeFromGOP registers a subscriber and returns the cached frames of the
// current GOP (starting with an IDR; empty if none cached). The subscription
// continues exactly after the last returned frame.
func (s *Stream) SubscribeFromGOP(buffer int) (*Subscription, []*Frame) {
	sub := &Subscription{C: make(chan Delivery, buffer)}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subs[sub] = struct{}{}
	return sub, slices.Clone(s.gop)
}

// Unsubscribe removes a subscriber and closes its channel.
func (s *Stream) Unsubscribe(sub *Subscription) {
	s.mu.Lock()
	if _, ok := s.subs[sub]; ok {
		delete(s.subs, sub)
		close(sub.C)
	}
	s.mu.Unlock()
}

// Publish delivers f to all subscribers.
func (s *Stream) Publish(f *Frame) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case f.IDR:
		s.gop = append(s.gop[:0:0], f)
	case f.NewSession || len(s.gop) >= maxGOP:
		s.gop = nil
	case len(s.gop) > 0:
		s.gop = append(s.gop, f)
	}
	for sub := range s.subs {
		select {
		case sub.C <- Delivery{Frame: f, Gap: sub.pending}:
			sub.pending = false
		default:
			sub.pending = true
		}
	}
}
