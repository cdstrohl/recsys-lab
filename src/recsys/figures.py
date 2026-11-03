"""Static figures rendered from `experiments.json` by the evaluation runner.

One figure for now: the headline metric by user history length, one panel per
evaluated split on a shared y-axis, mean with the user-bootstrap 95% interval.

Colors follow the model, never its rank (validated categorical slots 1-3, all
pairs, light surface). `random` is a dashed neutral reference line. Every series
is direct-labeled as well as in the legend, so identity never rests on color
alone, and the report's tables are the table view of the same numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8f8d87"  # `random`: a reference line, not a competing series
MODEL_COLORS = {
    "popularity": "#2a78d6",
    "category_affinity": "#eb6834",
    "tfidf": "#1baf7a",
}
# Later models take the remaining categorical slots in config order, never cycled.
EXTRA_COLORS = ("#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

SPLIT_TITLES = {
    "tune": "tune (models fitted on fit)",
    "holdout": "holdout (models refitted on fit + tune)",
}


def model_colors(models: Sequence[str]) -> dict[str, str]:
    extra = iter(EXTRA_COLORS)
    colors = {}
    for model in models:
        if model == "random":
            colors[model] = REFERENCE
        elif model in MODEL_COLORS:
            colors[model] = MODEL_COLORS[model]
        else:
            try:
                colors[model] = next(extra)
            except StopIteration:
                raise ValueError("more than 8 models; fold or facet instead of new hues") from None
    return colors


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def _spread(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart (keeping their order) so no two are closer than `gap`."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in pairwise(order):
        placed[cur] = max(placed[cur], placed[prev] + gap)
    return placed


def history_figure(experiments: Mapping[str, Any], path: Path) -> Path:
    """Headline metric by history-length bucket for every evaluated split."""
    config = experiments["config"]
    metric = f"ndcg@{max(config['ks'])}"
    label = metric.replace("ndcg", "nDCG")
    splits = experiments["splits"]
    models = list(next(iter(splits.values()))["models"])
    colors = model_colors(models)
    synthetic = "synthetic" in experiments["data_provenance"]

    fig = Figure(figsize=(5.2 * len(splits) + 1.2, 4.8), dpi=200, facecolor=SURFACE)
    axes = fig.subplots(1, len(splits), sharey=True, squeeze=False)[0]
    last_points: dict[str, tuple[float, float]] = {}
    for ax, (split_name, split) in zip(axes, splits.items(), strict=True):
        _style(ax)
        buckets = split["segments_by_history_length"]
        positions = range(len(buckets))
        for m, model in enumerate(models):
            dodge = (m - (len(models) - 1) / 2) * 0.08
            xs, ys, lows, highs = [], [], [], []
            for x, segment in zip(positions, buckets.values(), strict=True):
                entry = segment["models"][model]["metrics"][metric]
                if entry["mean"] is None:
                    continue
                xs.append(x + dodge)
                ys.append(entry["mean"])
                lows.append(entry["mean"] - (entry["ci_low"] or entry["mean"]))
                highs.append((entry["ci_high"] or entry["mean"]) - entry["mean"])
            reference = model == "random"
            ax.errorbar(
                xs,
                ys,
                yerr=[lows, highs],
                color=colors[model],
                linewidth=1.5,
                linestyle=(0, (4, 3)) if reference else "solid",
                elinewidth=1.0,
                capsize=0,
                marker="o",
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=model,
                zorder=2 if reference else 3,
            )
            if xs:
                last_points[model] = (xs[-1], ys[-1])
        ax.set_xticks(
            list(positions),
            [f"{name}\nn={seg['n_impressions']:,}" for name, seg in buckets.items()],
        )
        ax.set_xlim(-0.5, len(buckets) - 0.5)
        ax.set_title(
            SPLIT_TITLES.get(split_name, split_name), loc="left", fontsize=10, color=TEXT_PRIMARY
        )
        ax.set_xlabel("User History length (items)", fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_ylabel(f"{label} (mean, 95% CI)", fontsize=9, color=TEXT_SECONDARY)

    # Direct labels in one column right of the last panel's lines, nudged apart so none
    # collide; a thin leader in the series color ties each label to its line end.
    last = axes[-1]
    n_buckets = len(next(iter(splits.values()))["segments_by_history_length"])
    last.set_xlim(-0.5, n_buckets + 0.9)
    low, high = last.get_ylim()
    names = list(last_points)
    column = max((x for x, _ in last_points.values()), default=0.0) + 0.45
    ys = _spread([last_points[n][1] for n in names], gap=(high - low) * 0.075)
    for name, y in zip(names, ys, strict=True):
        last.annotate(
            name,
            xy=last_points[name],
            xytext=(column, y),
            fontsize=8.5,
            color=TEXT_PRIMARY,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": colors[name],
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )
    last.set_ylim(low, max(high, max(ys, default=high) + (high - low) * 0.04))

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
        ncol=len(models),
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
    )
    title = f"{label} by user History length"
    if synthetic:
        title += "  [SYNTHETIC FIXTURE DATA]"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.text(
        0.01,
        0.01,
        "Offline ranking metric on logged MIND-small-format impressions; not engagement or CTR."
        " Zero-history users: personalized models fall back to popularity, so those points"
        " coincide.",
        fontsize=7.5,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    return path
