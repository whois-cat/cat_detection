// Package config loads the shared config.yaml. streamhub reads only its own
// keys; other components read theirs. `${VAR}` references are expanded from
// the environment (e.g. RTSP credentials), and an unset variable is an error.
package config

import (
	"errors"
	"fmt"
	"os"
	"regexp"
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

// Camera is one camera.
type Camera struct {
	ID    string `yaml:"id"`
	Label string `yaml:"label"`
	RTSP  string `yaml:"rtsp"`
}

// Streamhub holds streamhub's settings.
type Streamhub struct {
	Listen     string     `yaml:"listen"`
	Recordings Recordings `yaml:"recordings"`
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

// Config is the whole config file; unknown top-level keys belong to other
// components and are ignored here.
type Config struct {
	// DataDir holds all runtime state; recordings go to <data_dir>/recordings.
	DataDir   string    `yaml:"data_dir"`
	Cameras   []Camera  `yaml:"cameras"`
	Streamhub Streamhub `yaml:"streamhub"`
}

// RecordingsDir returns the recordings root.
func (c *Config) RecordingsDir() string { return c.DataDir + "/recordings" }

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
			Listen: ":8090",
			Recordings: Recordings{
				SegmentTarget: Duration(10 * time.Second),
				Rescan:        Duration(time.Minute),
			},
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
