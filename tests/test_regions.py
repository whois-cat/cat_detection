from training.db import Box
from training.regions import box_in_ignore_region, parse_region_specs


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
