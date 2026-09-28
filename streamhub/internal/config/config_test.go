package config

import (
	"strings"
	"testing"
	"time"
)

func TestParse(t *testing.T) {
	t.Setenv("CAM_PASS", "s3cret")
	c, err := Parse([]byte(`
# comments may mention ${UNSET_IN_A_COMMENT}
data_dir: /data
cameras:
  - id: grey
    rtsp: rtsp://camera:${CAM_PASS}@192.168.0.213:554/stream1
streamhub:
  recordings:
    segment_target: 12s
    cachedir_tag: false
decider:
  anything: goes   # other components' keys are ignored
`))
	if err != nil {
		t.Fatal(err)
	}
	if c.Cameras[0].RTSP != "rtsp://camera:s3cret@192.168.0.213:554/stream1" {
		t.Errorf("rtsp = %q", c.Cameras[0].RTSP)
	}
	if time.Duration(c.Streamhub.Recordings.SegmentTarget) != 12*time.Second || *c.Streamhub.Recordings.CachedirTag {
		t.Errorf("recordings = %+v", c.Streamhub.Recordings)
	}
	if c.Streamhub.Listen != ":8090" || c.RecordingsDir() != "/data/recordings" {
		t.Errorf("defaults: listen %q, recordings %q", c.Streamhub.Listen, c.RecordingsDir())
	}
}

func TestParseErrors(t *testing.T) {
	for name, tc := range map[string]struct{ yaml, want string }{
		"unset env":    {"cameras: [{id: a, rtsp: 'rtsp://${NOPE_NOT_SET}@x'}]", "NOPE_NOT_SET"},
		"no cameras":   {"data_dir: x", "no cameras"},
		"bad id":       {"cameras: [{id: Grey, rtsp: 'rtsp://x'}]", "must match"},
		"duplicate id": {"cameras: [{id: a, rtsp: 'rtsp://x'}, {id: a, rtsp: 'rtsp://y'}]", "duplicate"},
		"bad url":      {"cameras: [{id: a, rtsp: 'http://x'}]", "rtsp://"},
		"bad duration": {"cameras: [{id: a, rtsp: 'rtsp://x'}]\nstreamhub: {recordings: {segment_target: 10}}", "missing unit"},
	} {
		_, err := Parse([]byte(tc.yaml))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want containing %q", name, err, tc.want)
		}
	}
}

func TestSizes(t *testing.T) {
	for in, want := range map[string]int64{"50GB": 50e9, "1.5GiB": 1.5 * (1 << 30), "123": 123, "10 kb": 10e3, "2TB": 2e12} {
		if got, err := parseSize(in); err != nil || got != want {
			t.Errorf("parseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	if _, err := parseSize("lots"); err == nil {
		t.Error("parseSize accepted garbage")
	}
	c, err := Parse([]byte("cameras: [{id: a, rtsp: 'rtsp://x'}]\npruner: {max_size: 20GB, keep_recent: 1h}"))
	if err != nil || c.Pruner.MaxSize != 20e9 || time.Duration(c.Pruner.KeepRecent) != time.Hour || !c.Pruner.DeleteUnprocessed {
		t.Errorf("pruner config: %+v %v", c.Pruner, err)
	}
}

func TestExampleConfig(t *testing.T) {
	for _, v := range []string{"CAM_GREY_PASSWORD", "CAM_BEIGE_PASSWORD", "CAM_BLACK_PASSWORD"} {
		t.Setenv(v, "x")
	}
	c, err := Load("../../../config.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Cameras) != 3 || c.Cameras[0].CV["rotate_deg"] != 90 || c.Streamhub.WebUIDir != "/webui" {
		t.Errorf("example config: %+v", c)
	}
}
