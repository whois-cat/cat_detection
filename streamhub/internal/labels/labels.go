// Package labels holds CV results and fans them out inside streamhub (to
// sidecar files, live viewers and the decider), and tracks which cameras have
// a CV worker attached.
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

// Bus fans results out to subscribers and tracks attached CV workers.
// Publishing never blocks; a slow subscriber misses results.
type Bus struct {
	mu   sync.Mutex
	subs map[chan Result]struct{}
	cv   map[string]int // camera -> attached workers
}

// NewBus returns an empty Bus.
func NewBus() *Bus {
	return &Bus{subs: map[chan Result]struct{}{}, cv: map[string]int{}}
}

// Subscribe returns a channel receiving all results.
func (b *Bus) Subscribe(buffer int) chan Result {
	ch := make(chan Result, buffer)
	b.mu.Lock()
	b.subs[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

// Unsubscribe removes and closes a subscription.
func (b *Bus) Unsubscribe(ch chan Result) {
	b.mu.Lock()
	if _, ok := b.subs[ch]; ok {
		delete(b.subs, ch)
		close(ch)
	}
	b.mu.Unlock()
}

// Publish delivers r to all subscribers.
func (b *Bus) Publish(r Result) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.subs {
		select {
		case ch <- r:
		default:
		}
	}
}

// Attach records a CV worker serving camera; the returned func detaches it.
func (b *Bus) Attach(camera string) (detach func()) {
	b.mu.Lock()
	b.cv[camera]++
	b.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.cv[camera]--
			b.mu.Unlock()
		})
	}
}

// CVActive reports whether a CV worker serves camera.
func (b *Bus) CVActive(camera string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cv[camera] > 0
}
