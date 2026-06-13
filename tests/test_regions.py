from training.db import Box
from training.regions import box_center_in_ignore_region, parse_region_specs


def test_parse_camera_ignore_region_and_match_box_center():
    regions = parse_region_specs(["grey:0.25,0.25,0.75,0.75"])
    box = Box(x=40, y=40, w=20, h=20, cat=None, score=0.9, track_id=None)

    assert box_center_in_ignore_region("grey", 100, 100, box, regions) is True
    assert box_center_in_ignore_region("beidge", 100, 100, box, regions) is False


def test_global_ignore_region_applies_to_any_camera():
    regions = parse_region_specs(["0.25,0.25,0.75,0.75"])
    box = Box(x=40, y=40, w=20, h=20, cat=None, score=0.9, track_id=None)

    assert box_center_in_ignore_region("beidge", 100, 100, box, regions) is True
