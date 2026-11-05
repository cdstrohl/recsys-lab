"""Run the baseline diagnostics (D-023). Reads MINDsmall_train only, never dev.

Holdout figures are read from the committed results/experiments.json as a
descriptive reference; nothing is re-scored on the holdout.

Usage:
    uv run python scripts/baseline_diagnostics.py --config configs/baseline_diagnostics.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recsys.baseline_diagnostics import (
    load_baseline_diagnostics_config,
    run_baseline_diagnostics,
    write_baseline_diagnostics,
)
from recsys.loaders import BEHAVIORS_FILE, NEWS_FILE, TRAIN_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_baseline_diagnostics_config(args.config)
    train_dir = config.base.data_dir / TRAIN_DIR
    missing = [p for p in (train_dir / NEWS_FILE, train_dir / BEHAVIORS_FILE) if not p.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}. See data/README.md.", file=sys.stderr)
        return 1

    results = run_baseline_diagnostics(config)
    for path in write_baseline_diagnostics(results, config):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
