#!/bin/sh
# Render mediamtx.yml.tpl → /tmp/mediamtx.yml with env-var substitution,
# then exec mediamtx.
#
# Why not just bind-mount a pre-rendered YAML? Because the path name has to
# match the detector's CAMERA_ID, the WebRTC host is install-specific, and
# the RTSP URL has credentials we don't want in source. One env-var-driven
# template centralises all three.
set -eu

: "${CAMERA_ID:?CAMERA_ID required}"
: "${RTSP_URL:?RTSP_URL required}"
: "${WEBRTC_HOST:?WEBRTC_HOST required}"

# sed-substitute the placeholders. Using | as the delimiter so URLs with
# slashes don't break us; escaping & in the URL just in case (rare).
RTSP_URL_ESCAPED=$(printf '%s\n' "$RTSP_URL" | sed 's/[&|]/\\&/g')
sed \
    -e "s|__CAMERA_ID__|${CAMERA_ID}|g" \
    -e "s|__RTSP_URL__|${RTSP_URL_ESCAPED}|g" \
    -e "s|__WEBRTC_HOST__|${WEBRTC_HOST}|g" \
    /mediamtx.yml.tpl > /tmp/mediamtx.yml

echo "[entrypoint] rendered /tmp/mediamtx.yml for camera=${CAMERA_ID}"
exec /mediamtx /tmp/mediamtx.yml
