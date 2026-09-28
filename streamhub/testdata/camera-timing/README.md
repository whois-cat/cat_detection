# Camera timing captures

Raw RTP/RTCP timing from the four Tapo C110 cameras, captured 2026-09-27 with
`rtpcap.py` (a minimal RTSP-over-TCP client — no ffmpeg, so timestamps are exactly
what the camera sent). 180 s per camera, all four in parallel, on the camera's second
RTSP session slot. Video payload is not included.

Used as replay input for streamhub's timeline tests (RTP unwrapping, wall-clock
anchoring, drift slewing, burst handling).

| Set | grey (hw 2.0, fw 1.5.4) | beige, black (hw 3.0, fw 1.3.1) | cam4 (hw 3.0, fw 1.3.1) |
|---|---|---|---|
| `2026-09-27-a` | 1920×1080 @15 — 50/50/100 ms cadence | 2304×1296 @15 | 2304×1296 @30, ~4% dropped frames |
| `2026-09-27-b` | 2304×1296 @30 — exact 33.3 ms grid | 2304×1296 @15 | 2304×1296 @30, ~4% dropped frames |

All: H.264 High, no B-frames, IDR every 60 frames. Cameras were in forced night mode.

## Files

`<camera>.rtp.csv.gz` — one row per RTP packet:

| Column | Meaning |
|---|---|
| `arrival_ns` | monotonic ns since capture start, when the packet was read |
| `wall_ns` | capture host wall-clock (Unix ns) at the same moment |
| `seq` | RTP sequence number |
| `rtp_ts` | RTP timestamp (32-bit, 90 kHz) |
| `marker` | RTP marker bit (1 = last packet of a frame) |
| `nal` | NAL unit type (`28` FU-A is resolved to the inner type; STAP-A as `stap:7+8`) |
| `start` | 1 if this packet starts a NAL unit |
| `size` | RTP payload size |

`<camera>.rtcp.csv.gz` — one row per RTCP Sender Report: `arrival_ns`, `wall_ns`,
`ntp_s` (camera's NTP time, Unix seconds), `rtp_ts`, `pkt_count`. Note: these
reports are inaccurate (NTP↔RTP mapping off by 0.4–0.6 s) — streamhub ignores RTCP.

## Scripts

- `rtpcap.py <name> <rtsp_url> <seconds> <outdir>` — capture (writes uncompressed
  CSVs, SDP and the raw Annex-B `.h264` stream).
- `analyze.py <dir> <camera>...` — per-camera summary: fps, frame spacing, GOP,
  arrival jitter, clock drift vs host, RTCP consistency.
