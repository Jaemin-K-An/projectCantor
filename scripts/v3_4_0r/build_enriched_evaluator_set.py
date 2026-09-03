"""Canonical entry point for target-model evaluator-set enrichment."""
from _common import require_external_window_pass
from elicit_compliance import main as _main


def main() -> None:
    require_external_window_pass()
    _main()


if __name__ == "__main__":
    main()
