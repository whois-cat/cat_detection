# mediamtx config TEMPLATE. Rendered at container start by entrypoint.sh,
# which substitutes the __PLACEHOLDER__ tokens below from env vars and
# writes the result to /tmp/mediamtx.yml. mediamtx itself doesn't do
# ${VAR} substitution in YAML; rather than rely on MTX_<UPPER> env
# overrides (which bake the path name into the env-var name — awkward for
# a dynamic path), we template it ourselves with a simple sed pass.
#
# Placeholders:
#   __CAMERA_ID__       — path name. Both mediamtx and the detector use
#                         this same string; recordings land under
#                         /recordings/<CAMERA_ID>/.
#   __RTSP_URL__        — camera RTSP source URL (with credentials).
#   __WEBRTC_HOST__     — host the browser dials for WebRTC ICE.

logLevel: info
logDestinations: [stdout]

# ----- WebRTC (live) -----
webrtc: yes
webrtcAddress: :8889
webrtcAllowOrigins: ['*']
webrtcEncryption: no
webrtcAdditionalHosts: ['__WEBRTC_HOST__']

# ----- Live HLS (we don't really use it — WebRTC is the live path — but harmless) -----
hls: yes
hlsAddress: :8888
hlsAllowOrigins: ['*']
hlsVariant: fmp4
hlsSegmentDuration: 1s

# ----- RTSP (republish, for the detector) -----
rtsp: yes
rtspAddress: :8554
rtspTransports: [tcp]

# ----- Disable other ingest -----
rtmp: no
srt: no

# ----- API & playback servers -----
api: yes
apiAddress: :9997
playback: yes
playbackAddress: :9996

# ----- Paths -----
paths:
  __CAMERA_ID__:
    source: __RTSP_URL__
    sourceProtocol: tcp
    sourceOnDemand: no

    # Recording to disk — fMP4 segments with PROGRAM-DATE-TIME. mediamtx
    # prunes anything older than recordDeleteAfter automatically.
    record: yes
    recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f
    recordFormat: fmp4
    recordSegmentDuration: 30s       # short → fine-grained pruning by detection presence
    recordPartDuration: 1s           # granularity inside a segment
    recordDeleteAfter: 720h          # 30-day absolute cap; pruner keeps last 24h whole,
                                     # detection-only for the older 29 days
