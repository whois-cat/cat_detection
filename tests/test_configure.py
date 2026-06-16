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
