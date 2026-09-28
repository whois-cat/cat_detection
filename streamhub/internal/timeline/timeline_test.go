package timeline

import (
	"compress/gzip"
	"encoding/csv"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

type frameSample struct {
	rtp     uint32
	arrival time.Time
}

// loadCapture groups captured RTP packets into frames (completed at the
// marker packet) with their arrival times.
func loadCapture(t *testing.T, path string) []frameSample {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	r := csv.NewReader(gz)
	header, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	rows, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	var frames []frameSample
	var wall0 time.Time
	for i, row := range rows {
		arrivalNS, _ := strconv.ParseInt(row[col["arrival_ns"]], 10, 64)
		if i == 0 {
			wallNS, _ := strconv.ParseInt(row[col["wall_ns"]], 10, 64)
			wall0 = time.Unix(0, wallNS-arrivalNS)
		}
		if row[col["marker"]] != "1" {
			continue
		}
		rtp, _ := strconv.ParseUint(row[col["rtp_ts"]], 10, 32)
		frames = append(frames, frameSample{rtp: uint32(rtp), arrival: wall0.Add(time.Duration(arrivalNS))})
	}
	return frames
}

type checkResult struct {
	minLag, maxWindowMinLag time.Duration
	// spacingViolations counts frames whose spacing differs from the camera's
	// by more than the slew allowance plus one tick of rounding.
	spacingViolations int
}

// run feeds frames through a Timeline and checks invariants. Checks that depend
// on convergence skip the first `settle` of media time.
func run(t *testing.T, frames []frameSample, settle time.Duration) checkResult {
	t.Helper()
	cfg := DefaultConfig()
	tl := New(cfg)
	pts := make([]int64, len(frames))
	for i, f := range frames {
		pts[i] = tl.Next(f.rtp, f.arrival)
		if i > 0 && pts[i] <= pts[i-1] {
			t.Fatalf("frame %d: pts not increasing: %d <= %d", i, pts[i], pts[i-1])
		}
	}

	res := checkResult{minLag: time.Hour}
	settleTicks := durationToTicks(settle)
	window := durationToTicks(cfg.Window)
	start := pts[0]
	windowStart, windowMin := int64(-1), time.Duration(time.Hour)
	for i := 1; i < len(frames); i++ {
		if pts[i]-start < settleTicks {
			continue
		}
		lag := frames[i].arrival.Sub(TicksToTime(pts[i]))
		res.minLag = min(res.minLag, lag)

		if windowStart < 0 {
			windowStart = pts[i]
		}
		windowMin = min(windowMin, lag)
		if pts[i]-windowStart >= window {
			res.maxWindowMinLag = max(res.maxWindowMinLag, windowMin)
			windowStart, windowMin = pts[i], time.Hour
		}

		rtpDelta := int64(int32(frames[i].rtp - frames[i-1].rtp))
		allowed := float64(rtpDelta)*cfg.MaxSlewPPM/1e6 + 1
		if float64(abs(pts[i]-pts[i-1]-rtpDelta)) > allowed {
			res.spacingViolations++
		}
	}
	return res
}

func abs(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func TestReplayCapturedCameras(t *testing.T) {
	paths, err := filepath.Glob("../../testdata/camera-timing/*/*.rtp.csv.gz")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no captures found: %v", err)
	}
	cfg := DefaultConfig()
	for _, p := range paths {
		name := filepath.Base(filepath.Dir(p)) + "/" + filepath.Base(p)
		t.Run(name, func(t *testing.T) {
			res := run(t, loadCapture(t, p), cfg.Warmup+cfg.Window)
			t.Logf("min lag %v, worst per-window min lag %v", res.minLag, res.maxWindowMinLag)
			// PTS never meaningfully after the frame arrived.
			if res.minLag < -5*time.Millisecond {
				t.Errorf("pts later than arrival by %v", -res.minLag)
			}
			// PTS tracks the earliest-arrival envelope in every window.
			if res.maxWindowMinLag > 20*time.Millisecond {
				t.Errorf("envelope tracking off by %v", res.maxWindowMinLag)
			}
			// Frame spacing is the camera's, adjusted at most by the slew bound.
			if res.spacingViolations > 0 {
				t.Errorf("%d frames adjusted beyond %v ppm", res.spacingViolations, cfg.MaxSlewPPM)
			}
		})
	}
}

// synthetic generates frames at fps with the camera clock drifting by driftPPM
// against the host and random burst delays of up to maxDelay.
func synthetic(n int, fps float64, driftPPM float64, maxDelay time.Duration, seed int64) []frameSample {
	rng := rand.New(rand.NewSource(seed))
	start := time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)
	rtp0 := uint32(rng.Int63())
	frames := make([]frameSample, n)
	var burstUntil int
	var burstDelay time.Duration
	for i := range frames {
		capture := time.Duration(float64(i) / fps * float64(time.Second))
		rtp := rtp0 + uint32(float64(i)/fps*ClockRate)
		host := start.Add(time.Duration(float64(capture) * (1 + driftPPM/1e6)))
		if i >= burstUntil && rng.Float64() < 0.05 {
			burstUntil = i + rng.Intn(int(fps))
			burstDelay = time.Duration(rng.Int63n(int64(maxDelay)))
		}
		delay := time.Duration(rng.Int63n(int64(3 * time.Millisecond)))
		if i < burstUntil {
			delay += burstDelay
		}
		frames[i] = frameSample{rtp: rtp, arrival: host.Add(delay)}
	}
	return frames
}

func TestSyntheticDrift(t *testing.T) {
	cfg := DefaultConfig()
	for _, drift := range []float64{-300, -50, 0, 170, 300} {
		frames := synthetic(30*60*15, 15, drift, time.Second, int64(drift))
		res := run(t, frames, cfg.Warmup+cfg.Window)
		t.Logf("drift %+.0f ppm: min lag %v, worst window min lag %v", drift, res.minLag, res.maxWindowMinLag)
		if res.minLag < -5*time.Millisecond || res.maxWindowMinLag > 20*time.Millisecond || res.spacingViolations > 0 {
			t.Errorf("drift %+.0f ppm: not tracked (min lag %v, window min lag %v)", drift, res.minLag, res.maxWindowMinLag)
		}
	}
}

func TestHostClockJumps(t *testing.T) {
	frames := synthetic(15*120, 15, 0, 100*time.Millisecond, 1)
	for i := 15 * 60; i < len(frames); i++ {
		frames[i].arrival = frames[i].arrival.Add(5 * time.Second)
	}
	tl := New(DefaultConfig())
	var prev int64
	for i, f := range frames {
		pts := tl.Next(f.rtp, f.arrival)
		if i > 0 && pts <= prev {
			t.Fatalf("frame %d: pts not increasing", i)
		}
		prev = pts
	}
	// Forward jump is stepped: the last frame lands near its arrival time.
	if lag := frames[len(frames)-1].arrival.Sub(TicksToTime(prev)); lag < 0 || lag > 50*time.Millisecond {
		t.Errorf("forward jump not followed: lag %v", lag)
	}

	// Backward jump is slewed, never breaking monotonicity.
	frames = synthetic(15*120, 15, 0, 100*time.Millisecond, 2)
	for i := 15 * 60; i < len(frames); i++ {
		frames[i].arrival = frames[i].arrival.Add(-5 * time.Second)
	}
	tl = New(DefaultConfig())
	for i, f := range frames {
		pts := tl.Next(f.rtp, f.arrival)
		if i > 0 && pts <= prev {
			t.Fatalf("frame %d: pts not increasing after backward jump", i)
		}
		prev = pts
	}
}

func TestReconnectKeepsIncreasing(t *testing.T) {
	a := synthetic(15*40, 15, 0, 50*time.Millisecond, 3)
	b := synthetic(15*40, 15, 0, 50*time.Millisecond, 4) // new random RTP base
	// Second session starts 2 s after the first one ended.
	shift := a[len(a)-1].arrival.Add(2 * time.Second).Sub(b[0].arrival)
	tl := New(DefaultConfig())
	var prev int64
	for i, f := range a {
		prev = tl.Next(f.rtp, f.arrival)
		_ = i
	}
	endOfA := prev
	tl.NewSession()
	for i, f := range b {
		pts := tl.Next(f.rtp, f.arrival.Add(shift))
		if pts <= prev {
			t.Fatalf("frame %d of new session: pts not increasing", i)
		}
		if i == 0 {
			if gap := TicksToDuration(pts - endOfA); gap < time.Second || gap > 3*time.Second {
				t.Errorf("reconnect gap %v, want ~2s", gap)
			}
		}
		prev = pts
	}
}

func TestTickConversions(t *testing.T) {
	tm := time.Date(2026, 9, 27, 23, 55, 18, 123456789, time.UTC)
	ticks := TimeToTicks(tm)
	if back := TicksToTime(ticks); back.Sub(tm).Abs() > time.Second/ClockRate {
		t.Errorf("roundtrip %v -> %v", tm, back)
	}
	if d := TicksToDuration(ClockRate * 3 / 2); d != 1500*time.Millisecond {
		t.Errorf("TicksToDuration = %v", d)
	}
}
