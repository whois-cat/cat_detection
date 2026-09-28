// Package mux builds fMP4 pieces for a single H.264 track at 90 kHz: init
// segments and fragments (moof+mdat). Shared by the recorder and live streaming
// so files on disk and live streams are byte-compatible.
package mux

import (
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/fmp4/seekablebuffer"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/mp4/codecs"

	"github.com/whois-cat/cat_detection/streamhub/internal/timeline"
)

const trackID = 1

// Init returns an init segment (ftyp+moov) for the given parameter sets.
func Init(sps, pps []byte) ([]byte, error) {
	init := fmp4.Init{Tracks: []*fmp4.InitTrack{{
		ID:        trackID,
		TimeScale: timeline.ClockRate,
		Codec:     &codecs.H264{SPS: sps, PPS: pps},
	}}}
	var buf seekablebuffer.Buffer
	if err := init.Marshal(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Sample converts an access unit to an fMP4 sample of the given duration (ticks).
func Sample(au [][]byte, idr bool, dur int64) (*fmp4.Sample, error) {
	payload, err := h264.AVCC(au).Marshal()
	if err != nil {
		return nil, err
	}
	return &fmp4.Sample{Duration: uint32(max(dur, 1)), IsNonSyncSample: !idr, Payload: payload}, nil
}

// Fragment returns a fragment holding samples, the first of which is at base (PTS ticks).
func Fragment(seq uint32, base int64, samples []*fmp4.Sample) ([]byte, error) {
	part := fmp4.Part{SequenceNumber: seq, Tracks: []*fmp4.PartTrack{{
		ID: trackID, BaseTime: uint64(base), Samples: samples,
	}}}
	var buf seekablebuffer.Buffer
	if err := part.Marshal(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
