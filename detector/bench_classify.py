"""CPU benchmark: sequential classify() vs classify_batch().

Needs a real OpenVINO classifier IR dir (cat_classifier.xml + classes.json).
Run inside the detector env, e.g.:

    python detector/bench_classify.py --model /opt/models/cat_classifier_openvino \
        --n 64 --batch 16 --repeat 5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from classifier import CatClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenVINO classifier IR dir")
    ap.add_argument("--n", type=int, default=64, help="crops per pass")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    clf = CatClassifier(args.model)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (args.size, args.size, 3), dtype=np.uint8)
             for _ in range(args.n)]

    def seq():
        return [clf.classify(c) for c in crops]

    def batched():
        out = []
        for i in range(0, len(crops), args.batch):
            out.extend(clf.classify_batch(crops[i:i + args.batch]))
        return out

    # Correctness: identical results regardless of path.
    assert seq() == batched(), "batch path disagrees with sequential"

    def timeit(fn):
        best = float("inf")
        for _ in range(args.repeat):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        return best

    t_seq = timeit(seq)
    t_batch = timeit(batched)
    print(f"crops={args.n} batch={args.batch} repeat={args.repeat}")
    print(f"  sequential: {t_seq * 1000:8.1f} ms  ({args.n / t_seq:6.1f} crops/s)")
    print(f"  batched   : {t_batch * 1000:8.1f} ms  ({args.n / t_batch:6.1f} crops/s)")
    print(f"  speedup   : {t_seq / t_batch:5.2f}x")


if __name__ == "__main__":
    main()
