package segment

import (
	"testing"
	"time"
)

func TestRoundTrip(t *testing.T) {
	start := time.Date(2026, 9, 27, 23, 55, 18, 123_456_789, time.UTC)
	p := FinalPath("grey", start, 10033*time.Millisecond+400*time.Microsecond)
	if want := "grey/2026-09-27/23/2026-09-27T23-55-18.123Z_10033ms.mp4"; p != want {
		t.Fatalf("FinalPath = %q, want %q", p, want)
	}
	info, err := Parse(p)
	if err != nil {
		t.Fatal(err)
	}
	if info.Camera != "grey" || !info.Start.Equal(start.Truncate(time.Millisecond)) || info.Duration != 10033*time.Millisecond {
		t.Errorf("Parse = %+v", info)
	}
	if got := PartPath("grey", start); got != "grey/2026-09-27/23/2026-09-27T23-55-18.123Z.part" {
		t.Errorf("PartPath = %q", got)
	}
}

func TestParseRejects(t *testing.T) {
	for _, p := range []string{
		"grey/2026-09-27/23/2026-09-27T23-55-18.123Z.part",
		"grey/2026-09-27/23/2026-09-27T23-55-18.123Z_10033ms.labels.jsonl",
		"grey/2026-09-27/22/2026-09-27T23-55-18.123Z_10033ms.mp4", // wrong hour dir
		"grey/2026-09-27T23-55-18.123Z_10033ms.mp4",
		"grey/2026-09-27/23/2026-09-27T23-55-18.123Z_10033.mp4",
		"grey/2026-09-27/23/garbage_10ms.mp4",
	} {
		if _, err := Parse(p); err == nil {
			t.Errorf("Parse(%q) succeeded", p)
		}
	}
}
