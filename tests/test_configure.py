import json

from tools import configure


def test_food_region_accepts_rect_mapping():
    cfg = {
        "cameras": [
            {
                "id": "grey",
                "label": "Grey",
                "food_region": {
                    "name": "bowl",
                    "rect": [0.13, 0.36, 0.26, 0.68],
                },
            }
        ]
    }

    rendered = json.loads(configure.render_cameras_json(cfg))

    assert rendered["cameras"][0]["food_region"] == {
        "name": "bowl",
        "points": [
            [0.13, 0.36],
            [0.26, 0.36],
            [0.26, 0.68],
            [0.13, 0.68],
        ],
    }


def test_detector_target_fps_rendered_with_default_and_override():
    default_compose = configure.render_compose({"cameras": [{"id": "grey"}]})
    tuned_compose = configure.render_compose({
        "cameras": [{"id": "grey", "detector_target_fps": 3.5}]
    })

    assert "DETECTOR_TARGET_FPS: 2.0" in default_compose
    assert "DETECTOR_TARGET_FPS: 3.5" in tuned_compose


def _feeder_cfg(feeder_extra: dict) -> dict:
    feeder = {
        "id": "feeder1",
        "api_base_url": "http://192.168.0.100",
        "serial_number": "ABC123",
        "allowed_cats": ["alisa"],
    }
    feeder.update(feeder_extra)
    return {"cameras": [{"id": "grey", "label": "Grey", "feeder": feeder}]}


def test_auto_refill_off_by_default():
    compose = configure.render_compose(_feeder_cfg({}))
    # No auto-refill keys → feeding disabled, defaults present (behaviour unchanged).
    assert 'FEED_ENABLED: "0"' in compose
    assert 'FEED_GRAIN_NUM: "1"' in compose
    assert 'FOOD_EMPTY_CONSECUTIVE: "2"' in compose
    assert 'FEED_MIN_INTERVAL_SEC: "1800"' in compose
    assert 'FEED_CONFIRM_TIMEOUT_SEC: "120"' in compose


def test_auto_refill_enabled_and_tuned():
    compose = configure.render_compose(_feeder_cfg({
        "feed_enabled": True,
        "feed_grain_num": 3,
        "food_empty_consecutive": 4,
        "feed_min_interval_sec": 900,
        "feed_confirm_timeout_sec": 60,
    }))
    assert 'FEED_ENABLED: "1"' in compose
    assert 'FEED_GRAIN_NUM: "3"' in compose
    assert 'FOOD_EMPTY_CONSECUTIVE: "4"' in compose
    assert 'FEED_MIN_INTERVAL_SEC: "900"' in compose
    assert 'FEED_CONFIRM_TIMEOUT_SEC: "60"' in compose
