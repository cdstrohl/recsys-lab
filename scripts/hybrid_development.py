"""Run E2 development: the leave-one-day-out hybrid on train days (D-025). Never reads dev.

Usage:
    uv run python scripts/hybrid_development.py --config configs/hybrid.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recsys.hybrid import load_hybrid_config, run_hybrid_development, write_hybrid
from recsys.loaders import BEHAVIORS_FILE, NEWS_FILE, TRAIN_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_hybrid_config(args.config)
    train_dir = config.base.data_dir / TRAIN_DIR
    missing = [p for p in (train_dir / NEWS_FILE, train_dir / BEHAVIORS_FILE) if not p.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}. See data/README.md.", file=sys.stderr)
        return 1

    results = run_hybrid_development(config)
    for path in write_hybrid(results, config):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
