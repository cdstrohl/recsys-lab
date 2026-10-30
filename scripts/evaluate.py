"""Evaluate every configured ranker on the tune and holdout splits.

Usage:
    uv run python scripts/evaluate.py --config configs/baseline.yaml
    uv run python scripts/evaluate.py --config configs/baseline.yaml --splits tune
    uv run python scripts/evaluate.py --config configs/baseline.yaml --lock-manifest

A holdout run requires the locked `eval/manifest.json` to match the data files.
Writes results/experiments.json, results/summary.json (holdout only) and the report.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from recsys.evaluate import (
    ManifestError,
    load_config,
    lock_manifest_from_config,
    metric_label,
    run,
    write_outputs,
)
from recsys.loaders import BEHAVIORS_FILE, DEV_DIR, NEWS_FILE, TRAIN_DIR
from recsys.splits import Split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[Split.TUNE.value, Split.HOLDOUT.value],
        help="override the config's splits",
    )
    parser.add_argument(
        "--lock-manifest",
        action="store_true",
        help="write the evaluation manifest (refuses to overwrite) and exit",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.splits:
        config = dataclasses.replace(config, splits=tuple(Split(s) for s in args.splits))
    missing = [
        p
        for d in (TRAIN_DIR, DEV_DIR)
        for p in (config.data_dir / d / NEWS_FILE, config.data_dir / d / BEHAVIORS_FILE)
        if not p.exists()
    ]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}. See data/README.md.", file=sys.stderr)
        return 1

    try:
        if args.lock_manifest:
            manifest = lock_manifest_from_config(config)
            print(f"Locked {config.manifest} ({manifest['manifest_version']}). Commit it.")
            return 0
        experiments = run(config)
    except ManifestError as exc:
        print(exc, file=sys.stderr)
        return 1

    headline = f"ndcg@{max(config.ks)}"
    for name, split in experiments["splits"].items():
        print(
            f"{name}: {split['n_impressions']} impressions, stats from {split['statistics_source']}"
        )
        for model, result in split["models"].items():
            m = result["metrics"][headline]
            print(f"  {model:<18} {metric_label(headline)} {m['mean']:.4f}  (n={m['n_eligible']})")
    for path in write_outputs(experiments, config):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
