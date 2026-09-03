"""Run the frozen normalized P0 grid on untouched behavioral CONFIRM."""
from __future__ import annotations

import argparse

from _common import RESULTS, read_json
from _run_symmetric import run_symmetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    freeze = read_json(RESULTS / "tables/p0_dose_grid_freeze.json")
    if freeze["status"] != "READY_FOR_CONFIRM":
        raise SystemExit("STOP: P0 dose grid was not frozen for confirmation")
    run_symmetric("D_beh_P0_confirm_335c", freeze["confirm_u_grid"], batch_size=args.batch)


if __name__ == "__main__":
    main()
