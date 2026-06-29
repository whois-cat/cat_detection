"""Render the multi-camera config files from cameras.yaml.

Generates (all marked with `# GENERATED ...` headers):
  - mediamtx/mediamtx.yml             one path per camera + global server config
  - docker-compose.cameras.yml        one `detector-<id>` service per camera
  - docker-compose.cameras.dev.yml    watchfiles override per detector (for `just dev`)
  - webui/nginx.conf                  per-camera detector upstreams + routing
  - webui/public/cameras.json         camera list the UI fetches at startup

Run via `just configure` (or `python tools/configure.py`) from the live2/
directory after editing cameras.yaml. Generated files are gitignored because
they are server-specific; rerun whenever cameras.yaml changes.

Networking model (important if you're chasing connectivity issues):
- mediamtx stays on the HOST network (network_mode: host) — needed so
  WebRTC ICE candidates resolve from the browser.
- Each detector runs on the DEFAULT bridge network with extra_hosts
  mapping host.docker.internal → host-gateway, so the detector reaches
  mediamtx at host.docker.internal:8554.
- webui (also on the bridge network) reaches each detector by container
  name on the docker DNS (detector-<id>). The webui's nginx fans out
  /detector/<id>/* and /detector-control/<id>/* to the right upstream.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CAMERAS_YAML = ROOT / "cameras.yaml"

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# Numeric range specs: (key, lo, hi, integer). Only keys actually present in the
# config are checked — defaults are trusted in-range. Confidences/fractions are
# [0,1]; counts/durations are non-negative. Catches out-of-range thresholds at
# `just configure` instead of at container start (or silently wrong behaviour).
CAMERA_RANGES = (
    ("detector_target_fps", 0, math.inf, False),
    ("yolo_conf", 0.0, 1.0, False),
    ("detector_unknown_conf", 0.0, 1.0, False),
    ("classifier_min_conf", 0.0, 1.0, False),       # legacy alias of the above
    ("classifier_pad_frac", 0.0, 1.0, False),
    ("blob_bright_threshold", 0, 255, True),
    ("blob_min_area", 0, math.inf, True),
    ("ignore_region_min_coverage", 0.0, 1.0, False),
    ("food_margin_frac", 0.0, 1.0, False),
    ("food_texture_weight", 0.0, 1.0, False),
    ("food_tiles", 1, math.inf, True),
    ("food_tex_thresh", 0.0, math.inf, False),
    ("food_empty_below", 0.0, 1.0, False),
    ("food_full_above", 0.0, 1.0, False),
    ("food_check_interval_sec", 0.0, math.inf, False),
    ("food_median_window", 1, math.inf, True),
    ("artificial_delay_ms", 0, math.inf, True),
)

FEEDER_RANGES = (
    ("classifier_min_conf", 0.0, 1.0, False),
    ("open_min_confidence", 0.0, 1.0, False),
    ("open_min_margin", 0.0, 1.0, False),
    ("door_close_timeout_sec", 0, math.inf, False),
    ("min_meal_sec", 0, math.inf, False),
    ("presence_window_sec", 0, math.inf, False),
    ("open_debounce_sec", 0, math.inf, False),
    ("multi_debounce_sec", 0, math.inf, False),
    ("display_text_interval", 0, math.inf, False),
    ("feed_grain_num", 1, math.inf, True),
    ("food_empty_consecutive", 1, math.inf, True),
    ("feed_min_interval_sec", 0, math.inf, False),
    ("feed_confirm_timeout_sec", 0, math.inf, False),
    ("feed_catchup_max", 0, math.inf, True),
)

VALID_ROTATE_DEG = (0, 90, 180, 270)


def _check_ranges(label: str, src: dict, specs) -> None:
    """sys.exit with a clear message if any present numeric key is out of range."""
    for key, lo, hi, integer in specs:
        if key not in src:
            continue
        v = src[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            sys.exit(f"{label}: {key} must be a number, got {v!r}")
        if integer and not float(v).is_integer():
            sys.exit(f"{label}: {key} must be a whole number, got {v!r}")
        if not (lo <= v <= hi):
            hi_s = "inf" if hi == math.inf else hi
            sys.exit(f"{label}: {key}={v} out of range [{lo}, {hi_s}]")


def _flat_points(raw) -> list[float]:
    if isinstance(raw, dict):
        coords = raw.get("rect", raw.get("points", raw.get("polygon")))
        if coords is None:
            raise ValueError("region dict must define rect, points, or polygon")
        return _flat_points(coords)
    if isinstance(raw, str):
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    if isinstance(raw, (list, tuple)):
        if raw and all(isinstance(v, (list, tuple)) for v in raw):
            return [float(x) for pair in raw for x in pair]
        return [float(v) for v in raw]
    raise ValueError(f"unsupported region coordinates: {raw!r}")


def _csv_coords(values: list[float]) -> str:
    return ",".join(f"{v:g}" for v in values)


def _decision_region_value(cam: dict):
    return cam.get("decision_polygon", cam.get("action_polygon", "0,0,1,1"))


def _region_coords(raw):
    if isinstance(raw, dict):
        return raw.get("rect", raw.get("points", raw.get("polygon")))
    return raw


def _region_name(cam: dict, raw, index: int) -> str:
    if isinstance(raw, dict):
        return str(raw.get("name") or f"{cam['id']}-ignore-{index}")
    return f"{cam['id']}-ignore-{index}"


def _is_soft_food_region_name(name: str) -> bool:
    tokens = str(name).lower().replace("-", "_").replace(" ", "_").split("_")
    return "bowl" in tokens or "food" in tokens


def _hard_ignore_region_values(cam: dict) -> list:
    return [
        raw
        for i, raw in enumerate(cam.get("ignore_regions", []) or [])
        if not _is_soft_food_region_name(_region_name(cam, raw, i))
    ]


def _food_region_value(cam: dict):
    explicit = cam.get("food_region")
    if explicit is not None:
        return explicit
    for i, raw in enumerate(cam.get("ignore_regions", []) or []):
        if _is_soft_food_region_name(_region_name(cam, raw, i)):
            return _region_coords(raw)
    return None


def _food_region_name(cam: dict) -> str:
    explicit = cam.get("food_region")
    if isinstance(explicit, dict):
        return str(explicit.get("name") or "bowl")
    if explicit is not None:
        return "bowl"
    for i, raw in enumerate(cam.get("ignore_regions", []) or []):
        if _is_soft_food_region_name(_region_name(cam, raw, i)):
            return _region_name(cam, raw, i)
    return "bowl"


def _detect_roi_value(cam: dict):
    return cam.get("detect_roi", "0,0,1,1")


def _polygon_from_value(raw) -> list[list[float]]:
    vals = _flat_points(raw)
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        points = [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]
    else:
        raise ValueError("region must be rect x0,y0,x1,y1 or an even polygon >= 3 points")
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"region point out of [0,1]: {(x, y)!r}")
    return points


def _polygon_env_value(raw) -> str:
    return _csv_coords([coord for point in _polygon_from_value(raw) for coord in point])


def _rect_from_value(raw) -> list[float]:
    vals = _flat_points(raw)
    if len(vals) != 4:
        raise ValueError("rect must be x0,y0,x1,y1")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"rect point out of [0,1]: {v!r}")
    return vals


def _ignore_regions_for_ui(cam: dict) -> list[dict]:
    out = []
    for i, raw in enumerate(_hard_ignore_region_values(cam)):
        name = _region_name(cam, raw, i)
        coords = _region_coords(raw)
        out.append({"name": name, "points": _polygon_from_value(coords)})
    return out


def load_config() -> dict:
    if not CAMERAS_YAML.exists():
        sys.exit(f"missing {CAMERAS_YAML} — copy cameras.yaml.example and edit")
    cfg = yaml.safe_load(CAMERAS_YAML.read_text())
    if not isinstance(cfg, dict) or "cameras" not in cfg:
        sys.exit("cameras.yaml must define a top-level `cameras:` list")
    if not cfg["cameras"]:
        sys.exit("cameras.yaml: at least one camera is required")
    seen_cam = set()
    seen_feeder = set()
    for cam in cfg["cameras"]:
        cid = cam.get("id")
        if not cid or not ID_RE.match(cid):
            sys.exit(f"invalid or missing camera id {cid!r} (must match {ID_RE.pattern})")
        if cid in seen_cam:
            sys.exit(f"duplicate camera id: {cid}")
        # mediamtx env-override path keys use '_' as the separator, so it
        # derives the key for MTX_PATHS_<KEY>_SOURCE by splitting on '_'. A
        # hyphenated path name can't be addressed this way (it would create a
        # mismatched key), and the RTSP source is now passed exclusively via
        # that env var — so camera ids feeding mediamtx must be hyphen-free.
        if "-" in cid:
            sys.exit(
                f"camera id {cid!r} must not contain '-': it becomes a mediamtx "
                "path overridden via MTX_PATHS_<ID>_SOURCE, whose key cannot "
                "carry hyphens"
            )
        seen_cam.add(cid)
        if not cam.get("rtsp"):
            sys.exit(f"camera {cid}: missing rtsp")

        _check_ranges(f"camera {cid}", cam, CAMERA_RANGES)
        rot = cam.get("rotate_deg", 0)
        if rot not in VALID_ROTATE_DEG:
            sys.exit(f"camera {cid}: rotate_deg must be one of "
                     f"{'/'.join(map(str, VALID_ROTATE_DEG))}, got {rot!r}")
        # Food hysteresis must be ordered or the bowl monitor can't latch state.
        empty_below = cam.get("food_empty_below", 0.30)
        full_above = cam.get("food_full_above", 0.55)
        if (isinstance(empty_below, (int, float)) and not isinstance(empty_below, bool)
                and isinstance(full_above, (int, float)) and not isinstance(full_above, bool)
                and empty_below >= full_above):
            sys.exit(f"camera {cid}: food_empty_below ({empty_below}) must be "
                     f"< food_full_above ({full_above})")

        feeder = cam.get("feeder")
        if feeder is not None:
            fid = feeder.get("id")
            if not fid or not ID_RE.match(fid):
                sys.exit(
                    f"camera {cid}: feeder.id {fid!r} invalid (must match {ID_RE.pattern})"
                )
            if fid in seen_feeder:
                sys.exit(f"camera {cid}: duplicate feeder id: {fid}")
            seen_feeder.add(fid)
            if not feeder.get("api_base_url"):
                sys.exit(f"camera {cid}: feeder.api_base_url is required")
            if not feeder.get("serial_number"):
                sys.exit(f"camera {cid}: feeder.serial_number is required")
            allowed = feeder.get("allowed_cats", [])
            if not isinstance(allowed, list) or not allowed:
                sys.exit(f"camera {cid}: feeder.allowed_cats must be a non-empty list")
            if not all(isinstance(c, str) and c.strip() for c in allowed):
                sys.exit(f"camera {cid}: feeder.allowed_cats must be a list of "
                         "non-empty strings")
            _check_ranges(f"camera {cid}: feeder", feeder, FEEDER_RANGES)
            dcs = feeder.get("dangerous_confusions", [])
            if not isinstance(dcs, list):
                sys.exit(f"camera {cid}: feeder.dangerous_confusions must be a list")
            for d in dcs:
                if not isinstance(d, dict) or "actual" not in d or "predicted" not in d:
                    sys.exit(f"camera {cid}: each feeder.dangerous_confusions entry "
                             "needs 'actual' and 'predicted'")
    return cfg


# ---- mediamtx ----------------------------------------------------------------

def mtx_source_env(cid: str) -> str:
    """Env var mediamtx reads to source path <cid> (see render_mediamtx)."""
    return f"MTX_PATHS_{cid.upper()}_SOURCE"


def render_mediamtx(cfg: dict) -> str:
    out = [
        "# GENERATED by tools/configure.py from cameras.yaml — do not edit by hand.",
        "logLevel: info",
        "logDestinations: [stdout]",
        "",
        "webrtc: yes",
        "webrtcAddress: :8889",
        "webrtcAllowOrigins: ['*']",
        "webrtcEncryption: no",
        f"webrtcAdditionalHosts: [{cfg.get('webrtc_host', '127.0.0.1')!r}]",
        "",
        "hls: yes",
        "hlsAddress: :8888",
        "hlsAllowOrigins: ['*']",
        "hlsVariant: fmp4",
        "hlsSegmentDuration: 1s",
        "",
        "rtsp: yes",
        "rtspAddress: :8554",
        "rtspTransports: [tcp]",
        "",
        "rtmp: no",
        "srt: no",
        "",
        "api: yes",
        "apiAddress: :9997",
        "playback: yes",
        "playbackAddress: :9996",
        "",
        "paths:",
    ]
    for cam in cfg["cameras"]:
        out += [
            f"  {cam['id']}:",
            # The RTSP source carries camera credentials and must NOT be written
            # to this committed file. mediamtx does no ${VAR} substitution in
            # YAML, so the source is injected at runtime via the env var below
            # (set from the gitignored secrets/cameras.env via docker-compose).
            f"    # source: set via {mtx_source_env(cam['id'])} (secrets/cameras.env)",
            "    sourceProtocol: tcp",
            "    sourceOnDemand: no",
            "    record: yes",
            "    recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f",
            "    recordFormat: fmp4",
            "    recordSegmentDuration: 30s",
            "    recordPartDuration: 1s",
            "    recordDeleteAfter: 720h",
        ]
    return "\n".join(out) + "\n"


# ---- docker-compose ----------------------------------------------------------

def render_compose(cfg: dict) -> str:
    services = []

    for cam in cfg["cameras"]:
        cid = cam["id"]
        food_region = _food_region_value(cam)
        env = {
            "CAMERA_ID":           cid,
            "RTSP_SOURCE":         f"rtsp://host.docker.internal:8554/{cid}",
            "DETECTOR_TYPE":       cam.get("detector_type", "blob"),
            "DETECTOR_TARGET_FPS": cam.get("detector_target_fps", 2.0),
            "BLOB_BRIGHT_THRESHOLD": cam.get("blob_bright_threshold", 240),
            "BLOB_MIN_AREA":       cam.get("blob_min_area", 500),
            "YOLO_WEIGHTS":        cam.get("yolo_weights", "/opt/models/yolov8n_int8_openvino_model/"),
            "YOLO_CONF":           cam.get("yolo_conf", 0.25),
            # Runtime classifier model comes from the shared read-only volume
            # (mounted below), not a baked image artifact. The `current` symlink
            # is switched by `just classifier-promote`; restart to pick it up.
            "CLASSIFIER_WEIGHTS":  cam.get("classifier_weights", "/opt/models/classifier/current"),
            "DETECTOR_UNKNOWN_CONF": cam.get(
                "detector_unknown_conf",
                cam.get("classifier_min_conf", 0.5),
            ),
            "CLASSIFIER_PAD_FRAC": cam.get("classifier_pad_frac", 0.15),
            "DETECT_ROI":          _csv_coords(_rect_from_value(_detect_roi_value(cam))),
            "IGNORE_REGIONS":      json.dumps(_hard_ignore_region_values(cam)),
            "IGNORE_REGION_MIN_COVERAGE": cam.get("ignore_region_min_coverage", 0.8),
            "ACTION_POLYGON":      _polygon_env_value(_decision_region_value(cam)),
            "FOOD_REGION":         _polygon_env_value(food_region) if food_region is not None else "",
            "FOOD_MARGIN_FRAC":    cam.get("food_margin_frac", 0.35),
            "FOOD_TEXTURE_WEIGHT": cam.get("food_texture_weight", 0.0),
            "FOOD_TILES":          cam.get("food_tiles", 8),
            "FOOD_TEX_THRESH":     cam.get("food_tex_thresh", 12.0),
            "FOOD_EMPTY_BELOW":    cam.get("food_empty_below", 0.30),
            "FOOD_FULL_ABOVE":     cam.get("food_full_above", 0.55),
            "FOOD_CHECK_INTERVAL_SEC": cam.get("food_check_interval_sec", 5),
            "FOOD_MEDIAN_WINDOW":  cam.get("food_median_window", 10),
            "FRAME_ROTATE_DEG":    cam.get("rotate_deg", 0),
            "EVENTS_DB":           "/data/events/events.db",
            "WS_PORT":             "8091",
            "CONTROL_PORT":        "8092",
            "ARTIFICIAL_DELAY_MS": cam.get("artificial_delay_ms", 0),
        }
        # Per-camera contrast calibration is optional: only emit the env vars
        # when present, so an uncalibrated camera leaves them unset and the
        # detector runs in calibration mode.
        if cam.get("food_empty_level") is not None:
            env["FOOD_EMPTY_LEVEL"] = cam["food_empty_level"]
        if cam.get("food_full_level") is not None:
            env["FOOD_FULL_LEVEL"] = cam["food_full_level"]
        env_lines = "\n".join(f"      {k}: {json.dumps(v)}" for k, v in env.items())
        services.append(f"""\
  detector-{cid}:
    build: ./detector
    container_name: detector-{cid}
    restart: unless-stopped
    depends_on: [mediamtx]
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
{env_lines}
    volumes:
      - ./data/events:/data/events
      # Shared, read-only classifier model volume. Mount the PARENT dir (not the
      # `current` symlink) so a promoted version switch is visible on restart.
      - ./models/classifier:/opt/models/classifier:ro
""")

        # Optional feeder service — generated only if feeder block present.
        feeder = cam.get("feeder")
        if feeder:
            fid = feeder["id"]
            allowed_cats = ",".join(feeder.get("allowed_cats", []))
            feeder_env = {
                "CAMERA_ID":               cid,
                "FEEDER_ID":               fid,
                "FEEDER_API_BASE_URL":     feeder["api_base_url"],
                "FEEDER_SERIAL_NUMBER":    feeder["serial_number"],
                "ALLOWED_CATS":            allowed_cats,
                # Optional dangerous confusions as a JSON string (feeder parses it).
                "DANGEROUS_CONFUSIONS":    json.dumps(feeder.get("dangerous_confusions", [])),
                "DOOR_CLOSE_TIMEOUT_SEC":  str(feeder.get("door_close_timeout_sec", 30)),
                "MIN_MEAL_SEC":            str(feeder.get("min_meal_sec", 10)),
                "PRESENCE_WINDOW_SEC":     str(feeder.get("presence_window_sec", 5)),
                "CLASSIFIER_MIN_CONF":     str(feeder.get("classifier_min_conf", 0.9)),
                "OPEN_MIN_CONFIDENCE":     str(feeder.get("open_min_confidence", 0)),
                "OPEN_MIN_MARGIN":         str(feeder.get("open_min_margin", 0)),
                "OPEN_DEBOUNCE_SEC":       str(feeder.get("open_debounce_sec", 3)),
                "MULTI_DEBOUNCE_SEC":      str(feeder.get("multi_debounce_sec", 2)),
                "DISPLAY_TEXT_INTERVAL":   str(feeder.get("display_text_interval", 2)),
                # Feeding mode: "empty_bowl" (default) reacts to the bowl monitor,
                # or "scheduled" feeds at fixed clock times. grain_num + the shared
                # journal apply to both; the journal volume is mounted below.
                "FEED_MODE":               str(feeder.get("feed_mode", "empty_bowl")),
                "FEED_GRAIN_NUM":          str(feeder.get("feed_grain_num", 1)),
                "FEED_JOURNAL_DB":         "/data/feed_journal/journal.db",
                # empty_bowl: OFF unless feed_enabled: true; needs a food_region.
                "FEED_ENABLED":            "1" if feeder.get("feed_enabled", False) else "0",
                "FOOD_EMPTY_CONSECUTIVE":  str(feeder.get("food_empty_consecutive", 2)),
                "FEED_MIN_INTERVAL_SEC":   str(feeder.get("feed_min_interval_sec", 1800)),
                "FEED_CONFIRM_TIMEOUT_SEC": str(feeder.get("feed_confirm_timeout_sec", 120)),
                # scheduled: fixed "HH:MM" slots; catch-up cap on restart/downtime.
                "FEED_TIMES":              ",".join(feeder.get("feed_times", [])),
                "FEED_TZ":                 str(feeder.get("feed_tz", "")),
                "FEED_CATCHUP_MAX":        str(feeder.get("feed_catchup_max", 2)),
            }
            feeder_env_lines = "\n".join(
                f"      {k}: {json.dumps(v)}" for k, v in feeder_env.items()
            )
            services.append(f"""\
  feeder-{fid}:
    build: ./feeder
    container_name: feeder-{fid}
    restart: unless-stopped
    depends_on: [detector-{cid}]
    environment:
{feeder_env_lines}
    volumes:
      - ./data/feed_journal:/data/feed_journal
""")

    return (
        "# GENERATED by tools/configure.py from cameras.yaml — do not edit.\n"
        "services:\n" + "\n".join(services)
    )


def render_compose_dev(cfg: dict) -> str:
    services = []
    for cam in cfg["cameras"]:
        cid = cam["id"]
        services.append(f"""\
  detector-{cid}:
    volumes:
      - ./detector:/app
      - ./data/events:/data/events
    command: ["python", "-u", "-m", "watchfiles", "--filter", "python", "python -u main.py", "/app"]
""")
    return (
        "# GENERATED by tools/configure.py from cameras.yaml — do not edit.\n"
        "services:\n" + "\n".join(services)
    )


# ---- nginx -------------------------------------------------------------------

def render_nginx(cfg: dict) -> str:
    # Hyphens aren't allowed in nginx upstream names; map id → safe name.
    def safe(cid: str) -> str:
        return cid.replace("-", "_")

    upstreams = "\n".join(
        f"upstream detector_{safe(cam['id'])}      {{ server detector-{cam['id']}:8091; }}\n"
        f"upstream detector_{safe(cam['id'])}_ctl  {{ server detector-{cam['id']}:8092; }}"
        for cam in cfg["cameras"]
    )
    detector_locations = "\n\n".join(f"""\
    location /detector/{cam['id']}/ {{
        proxy_pass http://detector_{safe(cam['id'])}/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }}
    location /detector-control/{cam['id']}/ {{
        proxy_pass http://detector_{safe(cam['id'])}_ctl/;
        proxy_http_version 1.1;
    }}""" for cam in cfg["cameras"])
    return f"""# GENERATED by tools/configure.py from cameras.yaml — do not edit.
{upstreams}

server {{
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    location / {{
        try_files $uri /index.html;
        add_header Cache-Control "no-store" always;
    }}

    # Per-camera detector REST + WS + control endpoints.
{detector_locations}

    # mediamtx (single instance, multiple paths) — path is part of the query string.
    location /whep/ {{
        proxy_pass http://host.docker.internal:8889/;
        proxy_http_version 1.1;
    }}
    location /recordings/ {{
        proxy_pass http://host.docker.internal:9996/;
        proxy_http_version 1.1;
        add_header Access-Control-Allow-Origin "*" always;
    }}
}}
"""


# ---- secrets (gitignored) ----------------------------------------------------

def render_secrets_env(cfg: dict) -> str:
    """Per-camera RTSP source (with credentials) as env vars for mediamtx.

    Written to secrets/cameras.env (gitignored) and loaded into the mediamtx
    container via docker-compose `env_file`. This is the ONLY place the RTSP
    credentials live on disk in the repo tree, and it never gets committed."""
    lines = [
        "# GENERATED by tools/configure.py from cameras.yaml — DO NOT COMMIT.",
        "# Contains camera RTSP credentials; loaded into the mediamtx container.",
    ]
    for cam in cfg["cameras"]:
        lines.append(f"{mtx_source_env(cam['id'])}={cam['rtsp']}")
    return "\n".join(lines) + "\n"


# ---- cameras.json (UI consumes) ---------------------------------------------

def render_cameras_json(cfg: dict) -> str:
    cams = []
    for cam in cfg["cameras"]:
        food_region = _food_region_value(cam)
        cams.append(
            {
                "id":    cam["id"],
                "label": cam.get("label", cam["id"].replace("-", " ").title()),
                "detect_roi": _rect_from_value(_detect_roi_value(cam)),
                "action_polygon": _polygon_from_value(_decision_region_value(cam)),
                "decision_polygon": _polygon_from_value(_decision_region_value(cam)),
                "food_region": (
                    {
                        "name": _food_region_name(cam),
                        "points": _polygon_from_value(food_region),
                    }
                    if food_region is not None
                    else None
                ),
                "ignore_regions": _ignore_regions_for_ui(cam),
                "ignore_region_min_coverage": cam.get("ignore_region_min_coverage", 0.8),
            }
        )
    return json.dumps({"cameras": cams}, indent=2) + "\n"


# ---- main --------------------------------------------------------------------

def main():
    cfg = load_config()
    targets = {
        ROOT / "mediamtx" / "mediamtx.yml":           render_mediamtx(cfg),
        ROOT / "docker-compose.cameras.yml":          render_compose(cfg),
        ROOT / "docker-compose.cameras.dev.yml":      render_compose_dev(cfg),
        ROOT / "webui" / "nginx.conf":                render_nginx(cfg),
        ROOT / "webui" / "public" / "cameras.json":   render_cameras_json(cfg),
    }
    for path, body in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"wrote {path.relative_to(ROOT)}")

    # Secrets are written separately: gitignored, never part of `targets`. chmod
    # 600 so credentials aren't world-readable.
    secrets_path = ROOT / "secrets" / "cameras.env"
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.write_text(render_secrets_env(cfg))
    secrets_path.chmod(0o600)
    print(f"wrote {secrets_path.relative_to(ROOT)} (gitignored — camera credentials)")


if __name__ == "__main__":
    main()
