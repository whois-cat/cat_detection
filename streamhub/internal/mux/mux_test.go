package mux

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4"
)

// TestWriteFixture builds a small segment (init + one fragment). With
// WRITE_WEBUI_FIXTURE=1 it also writes it as the webui's fMP4 parser test
// fixture, so both sides agree on the format.
func TestWriteFixture(t *testing.T) {
	data, err := os.ReadFile("../../testdata/tiny.h264")
	if err != nil {
		t.Fatal(err)
	}
	var au [][]byte
	var sps, pps []byte
	for _, n := range bytes.Split(data, []byte{0, 0, 1})[1:] {
		n = bytes.TrimRight(n, "\x00")
		switch h264.NALUType(n[0] & 0x1f) {
		case h264.NALUTypeAccessUnitDelimiter:
			if au != nil {
				goto done // first access unit only
			}
			au = [][]byte{}
		case h264.NALUTypeSPS:
			sps = n
			au = append(au, n)
		case h264.NALUTypePPS:
			pps = n
			au = append(au, n)
		default:
			au = append(au, n)
		}
	}
done:
	init, err := Init(sps, pps)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Sample(au, true, 6000)
	if err != nil {
		t.Fatal(err)
	}
	frag, err := Fragment(0, 160_000_000_000_000, []*fmp4.Sample{s})
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("WRITE_WEBUI_FIXTURE") == "" {
		return
	}
	out := filepath.Join("..", "..", "..", "webui", "src", "lib", "testdata", "segment.mp4")
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, append(init, frag...), 0o644); err != nil {
		t.Fatal(err)
	}
}
