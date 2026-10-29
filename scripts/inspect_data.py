"""Print an aggregate inventory of MIND-small files (no raw records).

Usage:
    uv run python scripts/inspect_data.py --data data/raw
    uv run python scripts/inspect_data.py --data tests/fixtures/mind_synthetic --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recsys.inventory import build_inventory, format_inventory
from recsys.loaders import BEHAVIORS_FILE, NEWS_FILE, TRAIN_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="directory holding MINDsmall_*")
    parser.add_argument("--json", type=Path, help="also write the inventory as JSON here")
    args = parser.parse_args(argv)

    train_dir = args.data / TRAIN_DIR
    missing = [p for p in (train_dir / NEWS_FILE, train_dir / BEHAVIORS_FILE) if not p.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}. See data/README.md.", file=sys.stderr)
        return 1

    inventory = build_inventory(args.data)
    print(format_inventory(inventory))
    if args.json:
        args.json.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
