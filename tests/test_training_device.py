"""Training picks the GPU only when asked for (auto) and torch sees one."""
from __future__ import annotations

import sys
import types

import pytest

from training import train_classifier, yolo_common


def _fake_torch(monkeypatch, cuda: bool) -> None:
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake)


@pytest.mark.parametrize("cuda, yolo, classifier", [(True, "0", "cuda"), (False, "cpu", "cpu")])
def test_auto_uses_the_gpu_only_when_cuda_is_available(monkeypatch, cuda, yolo, classifier):
    _fake_torch(monkeypatch, cuda)
    assert yolo_common.resolve_device("auto") == yolo
    assert train_classifier.resolve_device("auto") == classifier


def test_explicit_device_is_passed_through(monkeypatch):
    _fake_torch(monkeypatch, True)
    assert yolo_common.resolve_device("cpu") == "cpu"
    assert yolo_common.resolve_device("0,1") == "0,1"
    assert train_classifier.resolve_device("cpu") == "cpu"


def test_yolo_auto_without_torch_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    assert yolo_common.resolve_device("auto") == "cpu"


def test_device_defaults_to_auto():
    from training.yolo_evaluate import build_parser as eval_parser
    from training.yolo_train import build_parser as train_parser
    assert train_parser().parse_args(["--dataset", "d", "--weights", "w"]).device == "auto"
    assert eval_parser().parse_args(["--dataset", "d", "--model", "m"]).device == "auto"
