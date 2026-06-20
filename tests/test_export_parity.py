"""Model parity gate: strict on real crops, argmax-only on synthetic inputs.

_check_model_parity takes already-built callables (torch model + OpenVINO
compiled), so we drive it with tiny fakes — no openvino needed. The fakes look up
per-sample logits by an index encoded into the input tensor.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import export_classifier as ec  # detector/ is on sys.path via conftest


def _inputs(n):
    # Encode the sample index in the input so the fakes return per-sample logits.
    return [np.full((1, 3, 224, 224), i, dtype=np.float32) for i in range(n)]


class _FakeTorchModel:
    def __init__(self, logits):
        self._logits = logits
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, t):
        idx = int(t.reshape(-1)[0].item())
        return torch.tensor([self._logits[idx]], dtype=torch.float32)


class _FakeCompiled:
    def __init__(self, logits):
        self._logits = logits
        self._port = object()

    def output(self, i):
        assert i == 0
        return self._port

    def __call__(self, x):
        idx = int(np.asarray(x).reshape(-1)[0])
        return {self._port: np.array([self._logits[idx]], dtype=np.float32)}


def _run(torch_logits, ov_logits, *, synthetic):
    ec._check_model_parity(
        _FakeTorchModel(torch_logits),
        _FakeCompiled(ov_logits),
        _inputs(len(torch_logits)),
        synthetic=synthetic,
    )


# ---- synthetic mode: argmax-agreement only ----

def test_synthetic_passes_when_argmax_agrees_despite_prob_drift():
    # argmax matches on both samples; softmax probs differ well above PARITY_TOL
    # (1e-3) — synthetic mode must still PASS (no SystemExit).
    torch_logits = [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
    ov_logits = [[5.0, 0.0, 0.0], [0.0, 0.0, 4.0]]
    dp = float(np.max(np.abs(ec._softmax(np.array(torch_logits[0]))
                             - ec._softmax(np.array(ov_logits[0])))))
    assert dp > ec.PARITY_TOL  # the prob delta really is above the strict gate
    _run(torch_logits, ov_logits, synthetic=True)  # must NOT raise


def test_synthetic_fails_when_argmax_disagrees():
    # torch picks class 0, OpenVINO picks class 1 → different decision → FAIL.
    with pytest.raises(SystemExit):
        _run([[10.0, 0.0, 0.0]], [[0.0, 10.0, 0.0]], synthetic=True)


# ---- real mode: strict probability threshold preserved ----

def test_real_fails_on_prob_drift_even_when_argmax_agrees():
    # Same argmax, but max|Δprob| > PARITY_TOL → strict mode FAILs (unchanged).
    with pytest.raises(SystemExit):
        _run([[10.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]], synthetic=False)


def test_real_passes_when_probs_match():
    # Identical logits → Δprob == 0 and argmax agrees → PASS.
    _run([[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]], synthetic=False)
