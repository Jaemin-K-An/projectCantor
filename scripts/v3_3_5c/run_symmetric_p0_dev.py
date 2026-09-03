"""Run the preregistered symmetric normalized P0 dose grid on fresh DEV."""
from __future__ import annotations

import argparse

from _common import behavioral_protocol
from _run_symmetric import run_symmetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    protocol = behavioral_protocol()
    run_symmetric("D_beh_P0_dev_335c", protocol["candidate_u_grid"], batch_size=args.batch)


if __name__ == "__main__":
    main()
