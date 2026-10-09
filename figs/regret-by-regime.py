"""Policy regret by method and regime, packaging + pricing.

Regenerate: uv run --no-project --with matplotlib --with numpy python regret-by-regime.py
(run from this directory)
"""
import csv
from collections import defaultdict

import numpy as np
from orx_figstyle import PALETTE, MUTED, TEXT, figure_grid, panel_labels, save, use_style

METHODS = [("naive", "Naive comparison"), ("ab", "A/B test"),
           ("advisor", "Causal advisor")]
COLORS = {"naive": MUTED, "ab": PALETTE["blue"], "advisor": PALETTE["red"]}


def load(path, regime_of):
    cells = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            cells[(regime_of(row), row["method"])][int(row["rep"])].append(
                float(row["regret"]))
    # mean regret per replicate (over sub-cells), then mean + 95% CI over reps
    out = {}
    for key, reps in cells.items():
        per_rep = np.array([np.mean(v) for v in reps.values()])
        mean = 1000 * per_rep.mean()
        half = 1000 * 1.96 * per_rep.std(ddof=1) / np.sqrt(len(per_rep))
        out[key] = (mean, half)
    return out


def panel(ax, data, groups, ylabel):
    x = np.arange(len(groups))
    width = 0.9 / len(METHODS)
    for i, (m, label) in enumerate(METHODS):
        offset = (i - (len(METHODS) - 1) / 2) * width
        means = [data[(g, m)][0] for g, _ in groups]
        errs = [data[(g, m)][1] for g, _ in groups]
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=2,
                      label=label, color=COLORS[m], edgecolor="black",
                      linewidth=0.4, error_kw=dict(linewidth=0.8))
        ax.bar_label(bars, fmt="%.0f", fontsize=5, padding=1.5)
    ax.set_xticks(x, [name for _, name in groups])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


def main():
    use_style()
    pack = load("packaging-results.csv",
                lambda r: "low" if float(r["tau_x"]) < 4 else "high")
    price = load("pricing-results.csv",
                 lambda r: "inel" if float(r["b_p"]) < 0.2 else "el")

    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.42)
    panel(axes[0], pack,
          [("low", "Locking pays"),
           ("high", "Locking backfires")],
          "Regret ($ / 1k users / yr)")
    panel(axes[1], price,
          [("inel", "Raise pays"),
           ("el", "Raise loses money")],
          "Regret ($ / 1k signups / yr)")
    panel_labels(axes)
    axes[0].legend(loc="upper left", fontsize=6.5)
    save(fig, "regret-by-regime")


if __name__ == "__main__":
    main()
