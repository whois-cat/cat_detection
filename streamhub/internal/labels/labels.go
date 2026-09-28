// Package labels holds CV results and decider decisions, fans them out inside
// streamhub (to sidecar files, live viewers and the decider), and tracks which
// cameras have a CV worker attached.
package labels

import (
	"sync"
)

// Det is one detection. Coordinates are fractions of the camera frame (in
// camera orientation), so they don't depend on resolution or model rotation.
type Det struct {
	Box [4]float64 `msgpack:"box" json:"box"` // x, y, w, h
	// Score is the detector's confidence that the box is a cat at all.
	Score float64 `msgpack:"score" json:"score"`
	// Cats is the identity classifier's probability per known cat (sums to 1);
	// empty when the model has no identity classifier.
	Cats map[string]float64 `msgpack:"cats" json:"cats,omitempty"`
	// Emb is an optional embedding (little-endian float16), model-specific.
	Emb []byte `msgpack:"emb" json:"emb,omitempty"`
}

// Result is a CV worker's output for one frame. A frame with no detections
// still has a Result ("looked, found nothing").
type Result struct {
	Camera  string  `msgpack:"camera" json:"-"`
	PTS     int64   `msgpack:"pts" json:"pts"`
	Model   string  `msgpack:"-" json:"model"`
	Worker  string  `msgpack:"-" json:"worker"`
	InferMs float64 `msgpack:"infer_ms" json:"infer_ms"`
	Dets    []Det   `msgpack:"dets" json:"dets"`
}

// UnknownCat names a detection whose identity is uncertain; NoIdentity one
// from a model without an identity classifier.
const (
	UnknownCat = "unknown"
	NoIdentity = "cat"
	// identityMin is the probability below which a detection counts as
	// unknown for display purposes (the decider applies its own threshold).
	identityMin = 0.5
)

// TopCat returns the most likely identity of d for display.
func (d Det) TopCat() string {
	if len(d.Cats) == 0 {
		return NoIdentity
	}
	best, bestP := "", -1.0
	for name, p := range d.Cats {
		if p > bestP || (p == bestP && name < best) {
			best, bestP = name, p
		}
	}
	if bestP < identityMin {
		return UnknownCat
	}
	return best
}

// Decision is a decider's report (door state, reason, identity, …). Fields
// holds the whole record as sent, including camera and pts; streamhub only
// routes and stores it.
type Decision struct {
	Camera string
	PTS    int64
	Fields map[string]any
}

// Bus fans values out to subscribers. Publishing never blocks; a slow
// subscriber misses values.
type Bus[T any] struct {
	mu   sync.Mutex
	subs map[chan T]struct{}
}

// NewBus returns an empty Bus.
func NewBus[T any]() *Bus[T] {
	return &Bus[T]{subs: map[chan T]struct{}{}}
}

// Subscribe returns a channel receiving all values.
func (b *Bus[T]) Subscribe(buffer int) chan T {
	ch := make(chan T, buffer)
	b.mu.Lock()
	b.subs[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

// Unsubscribe removes and closes a subscription.
func (b *Bus[T]) Unsubscribe(ch chan T) {
	b.mu.Lock()
	if _, ok := b.subs[ch]; ok {
		delete(b.subs, ch)
		close(ch)
	}
	b.mu.Unlock()
}

// Publish delivers v to all subscribers.
func (b *Bus[T]) Publish(v T) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.subs {
		select {
		case ch <- v:
		default:
		}
	}
}

// CVTracker tracks which cameras have a CV worker attached.
type CVTracker struct {
	mu sync.Mutex
	cv map[string]int // camera -> attached workers
}

// Attach records a CV worker serving camera; the returned func detaches it.
func (t *CVTracker) Attach(camera string) (detach func()) {
	t.mu.Lock()
	if t.cv == nil {
		t.cv = map[string]int{}
	}
	t.cv[camera]++
	t.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			t.mu.Lock()
			t.cv[camera]--
			t.mu.Unlock()
		})
	}
}

// Active reports whether a CV worker serves camera.
func (t *CVTracker) Active(camera string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.cv[camera] > 0
}

// Channels bundles what flows between hub clients and streamhub's consumers.
type Channels struct {
	Results   *Bus[Result]
	Decisions *Bus[Decision]
	CV        *CVTracker

	mu   sync.Mutex
	last map[string]map[string]Decision // camera -> feeder -> latest decision
}

// NewChannels returns empty Channels.
func NewChannels() *Channels {
	return &Channels{Results: NewBus[Result](), Decisions: NewBus[Decision](), CV: &CVTracker{},
		last: map[string]map[string]Decision{}}
}

// PublishDecision remembers d as its feeder's latest and publishes it.
// Deciders report on change only, so the latest one is the current state.
func (c *Channels) PublishDecision(d Decision) {
	feeder, _ := d.Fields["feeder"].(string)
	c.mu.Lock()
	if c.last[d.Camera] == nil {
		c.last[d.Camera] = map[string]Decision{}
	}
	c.last[d.Camera][feeder] = d
	c.mu.Unlock()
	c.Decisions.Publish(d)
}

// LatestDecisions returns the latest decision of each feeder of camera.
func (c *Channels) LatestDecisions(camera string) []Decision {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Decision, 0, len(c.last[camera]))
	for _, d := range c.last[camera] {
		out = append(out, d)
	}
	return out
}
