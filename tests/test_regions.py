from training.db import Box
from training.regions import (
    box_in_ignore_region,
    load_ignore_regions_from_camera_config,
    parse_region_specs,
)


def test_parse_camera_ignore_region_and_drop_box_covered_by_region():
    regions = parse_region_specs(["grey:0.25,0.25,0.75,0.75"])
    box = Box(x=40, y=40, w=20, h=20, cat=None, score=0.9, track_id=None)

    assert box_in_ignore_region("grey", 100, 100, box, regions) is True
    assert box_in_ignore_region("beidge", 100, 100, box, regions) is False


def test_global_ignore_region_applies_to_any_camera():
    regions = parse_region_specs(["0.25,0.25,0.75,0.75"])
    box = Box(x=40, y=40, w=20, h=20, cat=None, score=0.9, track_id=None)

    assert box_in_ignore_region("beidge", 100, 100, box, regions) is True


def test_large_cat_box_over_ignore_region_is_kept():
    regions = parse_region_specs(["grey:0.25,0.25,0.75,0.75"])
    box = Box(x=10, y=10, w=80, h=80, cat=None, score=0.9, track_id=None)

    assert box_in_ignore_region("grey", 100, 100, box, regions) is False


def test_camera_config_bowl_region_is_not_a_hard_ignore(tmp_path):
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text(
        """
cameras:
  - id: grey
    ignore_regions:
      - name: feeder
        rect: [0.0, 0.0, 0.2, 1.0]
      - name: bowl
        rect: [0.2, 0.3, 0.4, 0.6]
""",
        encoding="utf-8",
    )

    regions = load_ignore_regions_from_camera_config(cfg)

    assert [r.name for r in regions["grey"]] == ["feeder"]
