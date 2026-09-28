// Package config loads the shared config.yaml. streamhub reads only its own
// keys; other components read theirs. `${VAR}` references are expanded from
// the environment (e.g. RTSP credentials), and an unset variable is an error.
package config

import (
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"go.yaml.in/yaml/v3"
)

// Duration is a time.Duration that unmarshals from strings like "10s".
type Duration time.Duration

// UnmarshalYAML implements yaml.Unmarshaler.
func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	v, err := time.ParseDuration(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	*d = Duration(v)
	return nil
}

// Camera is one camera. Its id is also its name everywhere (UI, paths).
type Camera struct {
	ID   string `yaml:"id"`
	RTSP string `yaml:"rtsp"`
	// CV is passed as-is to CV workers (e.g. rotate_deg, detect_roi).
	CV map[string]any `yaml:"cv"`
}

// Streamhub holds streamhub's settings.
type Streamhub struct {
	Listen string `yaml:"listen"`
	// HubListen is the address CV workers (and the decider) connect to.
	HubListen  string     `yaml:"hub_listen"`
	Recordings Recordings `yaml:"recordings"`
	// WebUIDir is the built webui to serve; empty disables it.
	WebUIDir string `yaml:"webui_dir"`
}

// Recordings holds recording settings.
type Recordings struct {
	SegmentTarget Duration `yaml:"segment_target"`
	// CachedirTag creates CACHEDIR.TAG in the recordings dir so backup tools
	// skip it. Opt-out.
	CachedirTag *bool `yaml:"cachedir_tag"`
	// Rescan is how often the recordings dir is rescanned for external changes.
	Rescan Duration `yaml:"rescan"`
}

// Pruner holds the pruner's settings.
type Pruner struct {
	// KeepRecent: recordings newer than this are never sparsified.
	KeepRecent Duration `yaml:"keep_recent"`
	// EventMargin: older recordings are kept this far around detections.
	EventMargin Duration `yaml:"event_margin"`
	// DeleteUnprocessed deletes old recordings CV never looked at.
	DeleteUnprocessed bool `yaml:"delete_unprocessed"`
	// MaxSize caps the recordings' total size, e.g. "50GB"; 0 disables.
	MaxSize ByteSize `yaml:"max_size"`
	// Interval between passes.
	Interval Duration `yaml:"interval"`
	DryRun   bool     `yaml:"dry_run"`
}

// ByteSize unmarshals sizes like "50GB", "500MiB" or a plain byte count.
type ByteSize int64

// UnmarshalYAML implements yaml.Unmarshaler.
func (b *ByteSize) UnmarshalYAML(n *yaml.Node) error {
	v, err := parseSize(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	*b = ByteSize(v)
	return nil
}

// Matched against the upper-cased input.
var sizeRe = regexp.MustCompile(`^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(I?)B?\s*$`)

func parseSize(s string) (int64, error) {
	m := sizeRe.FindStringSubmatch(strings.ToUpper(s))
	if m == nil {
		return 0, fmt.Errorf("bad size %q (want e.g. 50GB)", s)
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, err
	}
	base := 1000.0
	if m[3] == "I" {
		base = 1024
	}
	exp := 0
	if m[2] != "" {
		exp = strings.Index("KMGT", m[2]) + 1
	}
	return int64(v * math.Pow(base, float64(exp))), nil
}

// Config is the whole config file; unknown top-level keys belong to other
// components and are ignored here.
type Config struct {
	// DataDir holds all runtime state; recordings go to <data_dir>/recordings.
	DataDir   string    `yaml:"data_dir"`
	Cameras   []Camera  `yaml:"cameras"`
	Streamhub Streamhub `yaml:"streamhub"`
	Pruner    Pruner    `yaml:"pruner"`
}

// RecordingsDir returns the recordings root.
func (c *Config) RecordingsDir() string { return c.DataDir + "/recordings" }

// PinsFile returns the pinned-ranges file.
func (c *Config) PinsFile() string { return c.DataDir + "/pins.json" }

var (
	envRef   = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)
	cameraID = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)
)

// Load reads and validates a config file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse parses and validates config file contents.
func Parse(raw []byte) (*Config, error) {
	var missing []string
	expanded := envRef.ReplaceAllStringFunc(string(raw), func(ref string) string {
		name := envRef.FindStringSubmatch(ref)[1]
		v, ok := os.LookupEnv(name)
		if !ok {
			missing = append(missing, name)
		}
		return v
	})
	if len(missing) > 0 {
		return nil, fmt.Errorf("config references unset environment variables: %s", strings.Join(missing, ", "))
	}

	c := Config{
		DataDir: "data",
		Streamhub: Streamhub{
			Listen:    ":8090",
			HubListen: ":9000",
			Recordings: Recordings{
				SegmentTarget: Duration(10 * time.Second),
				Rescan:        Duration(time.Minute),
			},
		},
		Pruner: Pruner{
			KeepRecent:        Duration(3 * time.Hour),
			EventMargin:       Duration(30 * time.Second),
			DeleteUnprocessed: true,
			MaxSize:           50_000_000_000,
			Interval:          Duration(10 * time.Minute),
		},
	}
	if err := yaml.Unmarshal([]byte(expanded), &c); err != nil {
		return nil, err
	}
	if c.Streamhub.Recordings.CachedirTag == nil {
		t := true
		c.Streamhub.Recordings.CachedirTag = &t
	}
	return &c, c.validate()
}

func (c *Config) validate() error {
	var errs []error
	if len(c.Cameras) == 0 {
		errs = append(errs, errors.New("no cameras configured"))
	}
	seen := map[string]bool{}
	for i, cam := range c.Cameras {
		if !cameraID.MatchString(cam.ID) {
			errs = append(errs, fmt.Errorf("cameras[%d]: id %q must match %s", i, cam.ID, cameraID))
		}
		if seen[cam.ID] {
			errs = append(errs, fmt.Errorf("cameras[%d]: duplicate id %q", i, cam.ID))
		}
		seen[cam.ID] = true
		if !strings.HasPrefix(cam.RTSP, "rtsp://") {
			errs = append(errs, fmt.Errorf("camera %q: rtsp must be an rtsp:// URL", cam.ID))
		}
	}
	if c.Streamhub.Recordings.SegmentTarget <= 0 {
		errs = append(errs, errors.New("streamhub.recordings.segment_target must be positive"))
	}
	return errors.Join(errs...)
}
