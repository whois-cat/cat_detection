import numpy as np
import pytest

from training.train_classifier import (
    DangerousConfusion,
    Meta,
    build_episodes,
    check_split_leakage,
    confusion,
    dangerous_confusion_report,
    dangerous_from_confuse,
    decide_label,
    per_camera_confusion,
    per_class_f1,
    per_class_pr,
    per_group_accuracy,
    print_report,
    sample_indices_for_training,
    split_episodes,
    summarize_split,
    supported_macro_recall,
)


def test_episode_split_keeps_neighbours_together():
    metas = [
        Meta("alisa", "grey", 1_000),
        Meta("alisa", "grey", 2_000),
        Meta("chuzh", "grey", 200_000),
        Meta("chuzh", "grey", 201_000),
        Meta("ellie", "other", 1_000),
        Meta("ellie", "other", 2_000),
    ]

    episodes = build_episodes(metas, gap_ms=60_000)
    assert episodes == [[0, 1], [2, 3], [4, 5]]

    train, val, test = split_episodes(
        episodes,
        metas,
        val_frac=0.2,
        test_frac=0.2,
        required={"alisa", "chuzh", "ellie"},
        seed=7,
    )

    owners = {}
    for split_name, idxs in {"train": train, "val": val, "test": test}.items():
        for idx in idxs:
            owners[idx] = split_name

    for episode in episodes:
        assert len({owners[idx] for idx in episode}) == 1

    # The real split must pass the leakage guard.
    check_split_leakage(episodes, train, val, test)


def test_leakage_check_catches_split_violations():
    episodes = [[0, 1], [2, 3]]
    # crop 1 placed in both train and val → must raise.
    with pytest.raises(AssertionError):
        check_split_leakage(episodes, train_idx=[0, 1], val_idx=[1], test_idx=[2, 3])
    # episode [0,1] split across train/val → must raise.
    with pytest.raises(AssertionError):
        check_split_leakage(episodes, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    # clean split → no raise.
    check_split_leakage(episodes, train_idx=[0, 1], val_idx=[2, 3], test_idx=[])


def test_macro_recall_ignores_classes_absent_from_eval_split():
    # Middle class has no validation samples. This happens in weekly fine-tunes
    # when a cat is present only in replay memory, which is train-only.
    cm = np.array([
        [3, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ])

    assert supported_macro_recall(cm) == 1.0


def test_training_sampler_caps_duplicate_groups_and_keeps_suspicious():
    metas = [
        Meta(
            "alisa",
            "grey",
            i * 1_000,
            rowid=i,
            duplicate_group_id="visit-a",
            suspicious_score=1.0 if i == 4 else 0.0,
        )
        for i in range(10)
    ]
    episodes = build_episodes(metas, gap_ms=60_000)
    assert episodes == [list(range(10))]

    sampled = sample_indices_for_training(
        list(range(10)),
        episodes,
        metas,
        max_per_episode=5,
        max_per_duplicate_group=2,
        keep_suspicious_per_episode=1,
    )

    assert len(sampled) <= 5
    assert 0 in sampled          # first crop
    assert 9 in sampled          # last crop
    assert 4 in sampled          # hard/suspicious example
    check_split_leakage(episodes, train_idx=sampled, val_idx=[], test_idx=[])


def test_discard_and_unknown_are_not_classifier_labels():
    assert decide_label(None, None, "discard", False, 0.9) is None
    assert decide_label(None, None, "unknown", False, 0.9) is None
    assert decide_label("alisa", 0.99, None, False, 0.9) is None
    assert decide_label("alisa", 0.99, None, True, 0.9) == "alisa"


def test_summarize_split_reports_zero_overlap_for_clean_split():
    metas = [
        Meta("alisa", "grey", 1_000),
        Meta("alisa", "grey", 2_000),
        Meta("chuzh", "grey", 200_000),
        Meta("chuzh", "grey", 201_000),
        Meta("ellie", "other", 1_000),
        Meta("ellie", "other", 2_000),
    ]
    episodes = build_episodes(metas, gap_ms=60_000)
    # whole episodes to distinct splits → clean
    s = summarize_split(
        episodes, train_idx=[0, 1], val_idx=[2, 3], test_idx=[4, 5], metas=metas,
        identities=[f"row{i}" for i in range(len(metas))],
    )
    assert s["group_overlap_count"] == 0
    assert s["path_overlap_count"] == 0
    assert s["sample_counts"] == {"train": 2, "val": 2, "test": 2}
    assert s["group_counts"] == {"train": 1, "val": 1, "test": 1}
    assert s["class_dist"]["train"] == {"alisa": 2}


def test_summarize_split_detects_group_and_path_overlap():
    metas = [Meta("alisa", "grey", 1_000), Meta("alisa", "grey", 2_000)]
    episodes = [[0, 1]]
    # Same episode straddles train/val AND the same identity is reused.
    s = summarize_split(
        episodes, train_idx=[0], val_idx=[1], test_idx=[], metas=metas,
        identities=["dup", "dup"],
    )
    assert s["group_overlap_count"] == 1
    assert s["path_overlap_count"] >= 1


def test_confusion_matrix_is_dynamic_nxn():
    # Arbitrary class count → NxN matrix from the label list, never fixed 4x4.
    for n in (2, 3, 5, 7):
        classes = [f"cat_{i}" for i in range(n)]
        cm = confusion([0] * n, list(range(n)), len(classes))
        assert cm.shape == (n, n)


def test_per_class_precision_recall_f1():
    classes = ["cat_a", "cat_b", "cat_c"]
    # cat_a: 2 correct, cat_c: 2 true (1 correct, 1 -> cat_a)
    y_true = [0, 0, 2, 2]
    y_pred = [0, 0, 2, 0]
    cm = confusion(y_true, y_pred, len(classes))
    prec, rec = per_class_pr(cm)
    f1 = per_class_f1(prec, rec)
    # cat_a precision = 2/3, recall = 1.0 → F1 = 0.8
    assert round(float(f1[0]), 3) == 0.8
    out = print_report(cm, classes)
    assert set(out["f1"]) == set(classes)
    assert "macro_f1" in out


def test_dangerous_confusions_reported_generically(capsys):
    classes = ["cat_a", "cat_b", "cat_c"]
    # 2 true cat_b predicted cat_a (the dangerous direction); 1 true cat_a -> cat_b.
    y_true = [1, 1, 1, 0, 0]
    y_pred = [0, 0, 1, 1, 0]
    cm = confusion(y_true, y_pred, len(classes))
    dangerous = [DangerousConfusion(predicted="cat_a", actual="cat_b",
                                    reason="cat_a feeder must not open for cat_b")]

    total, details = dangerous_confusion_report(cm, classes, dangerous)
    assert total == 2
    assert details[0]["count"] == 2 and details[0]["actual"] == "cat_b"

    out = print_report(cm, classes, dangerous=dangerous)
    printed = capsys.readouterr().out
    assert out["dangerous_errors"] == 2
    assert "cat_b → cat_a: 2" in printed
    assert "feeder must not open" in printed
    assert "F1" in printed                 # per-class F1 is reported


def test_load_dangerous_confusions_from_yaml(tmp_path):
    from training.train_classifier import load_dangerous_confusions
    p = tmp_path / "danger.yaml"
    p.write_text(
        "dangerous_confusions:\n"
        "  - predicted: cat_a\n"
        "    actual: cat_b\n"
        "    reason: \"cat_a feeder must not open for cat_b\"\n",
        encoding="utf-8",
    )
    dcs = load_dangerous_confusions(p)
    assert len(dcs) == 1
    assert dcs[0].predicted == "cat_a" and dcs[0].actual == "cat_b"
    assert "feeder must not open" in dcs[0].reason


def test_load_dangerous_confusions_from_json_list(tmp_path):
    from training.train_classifier import load_dangerous_confusions
    p = tmp_path / "danger.json"
    p.write_text('[{"predicted":"x","actual":"y"}]', encoding="utf-8")  # JSON is valid YAML
    dcs = load_dangerous_confusions(p)
    assert len(dcs) == 1 and dcs[0].predicted == "x" and dcs[0].reason == ""


def test_dangerous_from_confuse_is_symmetric():
    dcs = dangerous_from_confuse({"cat_a", "cat_b"})
    pairs = {(d.actual, d.predicted) for d in dcs}
    assert pairs == {("cat_a", "cat_b"), ("cat_b", "cat_a")}
    assert dangerous_from_confuse({"cat_a"}) == []   # needs exactly two


def test_per_camera_and_per_group_breakdowns():
    classes = ["cat_a", "cat_b"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    cams = ["cam1", "cam1", "cam2", "cam2"]
    groups = [10, 10, 20, 20]

    pc = per_camera_confusion(y_true, y_pred, classes, cams)
    assert set(pc) == {"cam1", "cam2"}
    assert pc["cam1"]["accuracy"] == 0.5 and pc["cam2"]["accuracy"] == 1.0

    pg = per_group_accuracy(y_true, y_pred, groups)
    assert pg["n_groups"] == 2
    assert pg["worst"][0]["group"] == 10        # cam1 group is the worst

    # No metadata → empty, no crash.
    assert per_camera_confusion(y_true, y_pred, classes, []) == {}
    assert per_group_accuracy(y_true, y_pred, []) == {}
