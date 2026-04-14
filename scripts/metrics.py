from __future__ import annotations

import os

_DEFAULT_VM_URL = os.environ.get("VM_URL", "http://victoriametrics:8428")


def push_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
    vm_url: str = _DEFAULT_VM_URL,
) -> None:
    """Push a single metric to VictoriaMetrics in Prometheus text format."""
    import requests

    label_str = ""
    if labels:
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        label_str = f"{{{parts}}}"

    line = f"{name}{label_str} {value}\n"

    try:
        requests.post(
            f"{vm_url}/api/v1/import/prometheus",
            data=line.encode(),
            headers={"Content-Type": "text/plain"},
            timeout=3,
        )
    except requests.RequestException:
        pass
