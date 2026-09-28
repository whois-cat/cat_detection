package media

import "testing"

func TestGOPCacheAndGaps(t *testing.T) {
	s := NewStream()
	for i := range 5 {
		s.Publish(&Frame{PTS: int64(i), IDR: i == 0 || i == 3})
	}
	sub, gop := s.SubscribeFromGOP(1)
	if len(gop) != 2 || gop[0].PTS != 3 || gop[1].PTS != 4 {
		t.Fatalf("gop = %v", gop)
	}
	s.Publish(&Frame{PTS: 5})
	s.Publish(&Frame{PTS: 6}) // buffer full: dropped
	if d := <-sub.C; d.PTS != 5 || d.Gap {
		t.Fatalf("got %+v", d)
	}
	s.Publish(&Frame{PTS: 7})
	if d := <-sub.C; d.PTS != 7 || !d.Gap {
		t.Fatalf("want gap-marked frame 7, got pts %d gap %v", d.PTS, d.Gap)
	}
	// Cached GOP is not affected by later appends to the returned slice.
	gop[0] = nil
	if _, gop2 := s.SubscribeFromGOP(1); gop2[0] == nil || len(gop2) != 5 {
		t.Fatalf("gop2 = %v", gop2)
	}
	s.Unsubscribe(sub)
	if _, ok := <-sub.C; ok {
		t.Fatal("channel not closed")
	}
}
