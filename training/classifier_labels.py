"""Label policy: turn (detector label, human review) into a training label."""
from __future__ import annotations


DROP_LABELS = {"discard", "unknown"}


def decide_label(det_label, det_conf, human, trust_classifier, trust_conf):
    """Final training label for one crop, or None to drop it.

    Default (trust_classifier=False): ONLY human labels are used. With
    --trust-classifier, an unreviewed crop may also use the existing classifier's
    label when its cat_score >= trust_conf. discard/unknown are always dropped.
    """
    if human is not None:
        return None if human in DROP_LABELS else human
    if not trust_classifier:
        return None                      # human-only by default
    if not det_label or det_label in DROP_LABELS:
        return None
    if det_conf is not None and det_conf >= trust_conf:
        return det_label
    return None


def require_min_classes(classes: list[str]) -> None:
    """Refuse to train an identity classifier on fewer than two labeled classes.

    A single class yields a degenerate model that always predicts that class —
    useless, yet still saveable/promotable. Fail loudly before model creation.
    Class names are reported dynamically (no assumptions about which labels).
    """
    if len(classes) < 2:
        raise SystemExit(
            "need at least 2 labeled classes to train the identity classifier; "
            f"found classes: {classes}. Review more crops (just label-review) or "
            "check your drop labels / review labels."
        )
