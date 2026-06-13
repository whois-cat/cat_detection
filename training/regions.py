"""Ignore-region helpers for detector false-positive cleanup.

Regions are expressed in camera-normalized coordinates [0..1]. A detection is
ignored when its box center falls inside any configured region for its camera
or a global "*" region.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .db import Box


@dataclass(frozen=True, slots=True)
class IgnoreRegion:
    name: str
    points: tuple[tuple[float, float], ...]


def _point_in_polygon(x: float, y: float, poly: Iterable[tuple[float, float]]) -> bool:
    pts = list(poly)
    inside = False
    j = len(pts) - 1
    for i, (xi, yi) in enumerate(pts):
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xc:
                inside = not inside
        j = i
    return inside


def _flat_points(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    if isinstance(raw, (list, tuple)):
        if raw and all(isinstance(v, (list, tuple)) for v in raw):
            return [float(x) for pair in raw for x in pair]
        return [float(v) for v in raw]
    raise ValueError(f"unsupported region coordinates: {raw!r}")


def region_from_value(raw, *, default_name: str = "ignore") -> IgnoreRegion:
    name = default_name
    coords = raw
    if isinstance(raw, Mapping):
        name = str(raw.get("name") or default_name)
        coords = raw.get("rect", raw.get("points", raw.get("polygon")))
    vals = _flat_points(coords)
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        points = tuple((vals[i], vals[i + 1]) for i in range(0, len(vals), 2))
    else:
        raise ValueError("region must be rect x0,y0,x1,y1 or an even polygon >= 3 points")
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"region point out of [0,1]: {(x, y)!r}")
    return IgnoreRegion(name=name, points=tuple(points))


def parse_region_specs(specs: Iterable[str]) -> dict[str, list[IgnoreRegion]]:
    """Parse CLI specs: 'camera:x0,y0,x1,y1' or global 'x0,y0,x1,y1'."""
    out: dict[str, list[IgnoreRegion]] = {}
    for spec in specs:
        camera = "*"
        coords = spec
        head, sep, tail = spec.partition(":")
        if sep and not head.replace(".", "", 1).isdigit():
            camera = head
            coords = tail
        out.setdefault(camera, []).append(region_from_value(coords, default_name="cli"))
    return out


def load_ignore_regions_from_camera_config(path: Path) -> dict[str, list[IgnoreRegion]]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, list[IgnoreRegion]] = {}
    for cam in cfg.get("cameras", []):
        cid = cam.get("id")
        if not cid:
            continue
        regions = []
        for i, raw in enumerate(cam.get("ignore_regions", []) or []):
            regions.append(region_from_value(raw, default_name=f"{cid}-ignore-{i}"))
        if regions:
            out[cid] = regions
    return out


def merge_regions(*maps: dict[str, list[IgnoreRegion]]) -> dict[str, list[IgnoreRegion]]:
    out: dict[str, list[IgnoreRegion]] = {}
    for mapping in maps:
        for camera, regions in mapping.items():
            out.setdefault(camera, []).extend(regions)
    return out


def box_center_in_ignore_region(
    camera_id: str,
    frame_w: int,
    frame_h: int,
    box: Box,
    regions_by_camera: dict[str, list[IgnoreRegion]],
) -> bool:
    if frame_w <= 0 or frame_h <= 0:
        return False
    regions = [
        *regions_by_camera.get("*", []),
        *regions_by_camera.get(camera_id, []),
    ]
    if not regions:
        return False
    cx = (box.x + box.w * 0.5) / frame_w
    cy = (box.y + box.h * 0.5) / frame_h
    return any(_point_in_polygon(cx, cy, region.points) for region in regions)


def regions_to_jsonable(regions_by_camera: dict[str, list[IgnoreRegion]]) -> dict:
    return {
        camera: [
            {"name": region.name, "points": [list(p) for p in region.points]}
            for region in regions
        ]
        for camera, regions in sorted(regions_by_camera.items())
    }
