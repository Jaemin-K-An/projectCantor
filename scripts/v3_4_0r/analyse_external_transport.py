"""Summarize independent transport discrimination and fixed-window applicability."""
from __future__ import annotations

from _common import RESULTS, read_json, write_json


def main() -> None:
    sensor = read_json(RESULTS / "tables/sensor_transfer.json")
    window = read_json(RESULTS / "tables/external_window.json")
    payload = {
        "sensor_transport": sensor["transport_verdict"],
        "sensor_scope": sensor["sensor_scope"],
        "auroc": sensor["auroc"],
        "auroc_ci95": sensor["auroc_ci95"],
        "balanced_accuracy_at_zero": sensor["balanced_accuracy_at_zero"],
        "fixed_W_coverage": window["coverage"],
        "fixed_W_gate": window["coverage_min"],
        "window_verdict": window["verdict"],
        "overall_transport_state": window["verdict"] if not window["passed"] else sensor["transport_verdict"],
        "sensor_retrained": False,
        "W_recalibrated": False,
    }
    write_json(RESULTS / "tables/external_transport.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
