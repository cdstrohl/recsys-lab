"""Exploratory holdout run of the frozen E2 hybrids (D-022, D-027).

Usage:
    # 1. Train only: fit and record the frozen weights and plan, then commit the file.
    uv run python scripts/hybrid_holdout.py --config configs/hybrid_holdout.yaml --freeze
    # 2. Score MIND-small dev with those weights (refuses unless step 1 is committed).
    uv run python scripts/hybrid_holdout.py --config configs/hybrid_holdout.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from recsys.hybrid_holdout import (
    freeze_hybrids,
    load_holdout_config,
    run_exploratory_holdout,
    write_freeze,
    write_holdout,
)


def _committed(path: Path) -> bool:
    """True when `path` is tracked by git and has no uncommitted changes."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True)
    if tracked.returncode != 0:
        return False
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)], capture_output=True, text=True
    )
    return status.returncode == 0 and not status.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--freeze", action="store_true", help="fit and write the frozen weights (train only)"
    )
    args = parser.parse_args(argv)
    config = load_holdout_config(args.config)

    if args.freeze:
        if config.freeze.exists():
            print(
                f"{config.freeze} already exists; a freeze is never overwritten.", file=sys.stderr
            )
            return 1
        print(f"wrote {write_freeze(freeze_hybrids(config), config)}")
        return 0

    if not _committed(config.freeze):
        print(
            f"{config.freeze} must be committed, unchanged, before the holdout is scored.",
            file=sys.stderr,
        )
        return 1
    for path in write_holdout(run_exploratory_holdout(config), config):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
