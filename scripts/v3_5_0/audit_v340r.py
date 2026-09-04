"""Phase 0: prove that all historical V3.4.0/V3.4.0R trees are untouched."""
from __future__ import annotations

import subprocess

from _common import (ACTUATOR_SHA256, BASE_COMMIT, CONFIG, RESULTS,
                     SENSOR_SHA256, ROOT, sha256, write_json)

PRESERVED = [
    "results/v3_4_0", "results/v3_4_0r", "configs/v3_4_0", "configs/v3_4_0r",
    "docs/v3_4_0", "docs/v3_4_0r",
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    trees = {}
    for path in PRESERVED:
        base = git("rev-parse", f"{BASE_COMMIT}:{path}")
        head = git("rev-parse", f"HEAD:{path}")
        trees[path] = {"base_tree": base, "head_tree": head, "identical": base == head}
    sensor = sha256(ROOT / "results/v3_4_0/cache/sensor_w.npy")
    actuator = sha256(ROOT / "results/v3_3_5a/cache/v_p0.npy")
    passed = all(v["identical"] for v in trees.values()) and sensor == SENSOR_SHA256 and actuator == ACTUATOR_SHA256
    payload = {
        "base_commit": BASE_COMMIT,
        "preserved_paths": trees,
        "sensor_sha256": sensor,
        "sensor_hash_exact": sensor == SENSOR_SHA256,
        "actuator_sha256": actuator,
        "actuator_hash_exact": actuator == ACTUATOR_SHA256,
        "passed": passed,
        "verdict": "AUDIT1_HISTORICAL_IMMUTABLE" if passed else "AUDIT2_HISTORICAL_MUTATION",
    }
    write_json(RESULTS / "tables/v340r_audit.json", payload)
    if not passed:
        raise SystemExit("historical immutability audit failed")
    print(payload["verdict"])


if __name__ == "__main__":
    main()
