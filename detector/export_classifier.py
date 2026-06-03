"""Build-time script: export cat_classifier.pt → OpenVINO IR + classes.json.

Run once during Docker image build:
  python export_classifier.py \
    --pt /app/models/cat_classifier.pt \
    --out /opt/models/cat_classifier_openvino/

Requires torch + torchvision + openvino (build-time only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def export(pt_path: Path, out_dir: Path) -> None:
    import openvino as ov
    import torch
    import torch.nn as nn
    from torchvision import models

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    ov_model = ov.convert_model(model, example_input=dummy)

    xml_path = out_dir / "cat_classifier.xml"
    ov.save_model(ov_model, str(xml_path))
    print(f"[export] OpenVINO IR → {xml_path}", flush=True)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(class_names), encoding="utf-8")
    print(f"[export] classes → {classes_path} : {class_names}", flush=True)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export(Path(args.pt), Path(args.out))


if __name__ == "__main__":
    main()
