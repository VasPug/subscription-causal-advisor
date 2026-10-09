"""What each method believes vs the truth. Regenerate:
uv run --no-project --with matplotlib --with numpy python estimates-vs-truth.py
(from figs/; reads the run CSVs copied here)
"""
import csv
from collections import defaultdict

import numpy as np
from orx_figstyle import PALETTE, MUTED, TEXT, figure_grid, panel_labels, save, use_style

COLORS = {"naive": MUTED, "advisor": PALETTE["red"], "true": PALETTE["blue"]}


def rows(path):
    with open(path) as f:
        yield from csv.DictReader(f)


def mean_ci(vals):
    v = np.array(vals)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main():
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.42)

    # (a) packaging: estimated $ value of feature access vs hidden-bias dial
    est = defaultdict(lambda: defaultdict(list))
    truth = defaultdict(list)
    for r in rows("packaging-results.csv"):
        if r["method"] in ("naive", "advisor") and r["tau_hat"] != "nan":
            est[r["method"]][float(r["conf"])].append(float(r["tau_hat"]))
        if r["method"] == "naive":
            truth[float(r["conf"])].append(float(r["tau_true"]))
    confs = sorted(truth)
    ax = axes[0]
    for m, label in [("naive", "Naive comparison"), ("advisor", "Advisor")]:
        mu = [mean_ci(est[m][c])[0] for c in confs]
        hw = [mean_ci(est[m][c])[1] for c in confs]
        ax.errorbar(confs, mu, yerr=hw, marker="o", capsize=2,
                    color=COLORS[m if m != "naive" else "naive"], label=label)
    tmu = [np.mean(truth[c]) for c in confs]
    ax.plot(confs, tmu, "--", color=COLORS["true"], label="True value")
    ax.set_xlabel("Hidden-bias dial")
    ax.set_ylabel("Estimated feature value (\\$/user/yr)")
    ax.set_xticks(confs)
    ax.legend(fontsize=6.5, loc="upper left")

    # (b) pricing: elasticity error of each method, with and without a quality trend
    err = defaultdict(lambda: defaultdict(list))
    for r in rows("pricing-results.csv"):
        if r["has_var"] == "1" and r["method"] in ("naive", "advisor") \
                and r["b_hat"] not in ("nan", ""):
            err[r["method"]][float(r["drift"])].append(
                float(r["b_hat"]) - float(r["b_p"]))
    ax = axes[1]
    drifts = sorted(err["naive"])
    x = np.arange(len(drifts))
    w = 0.35
    for i, (m, label) in enumerate([("naive", "Naive comparison"), ("advisor", "Advisor")]):
        mu = [mean_ci(err[m][d])[0] for d in drifts]
        hw = [mean_ci(err[m][d])[1] for d in drifts]
        bars = ax.bar(x + (i - 0.5) * w, mu, w, yerr=hw, capsize=2,
                      color=COLORS[m if m != "naive" else "naive"], label=label,
                      edgecolor="black", linewidth=0.4,
                      error_kw=dict(linewidth=0.8))
        ax.bar_label(bars, fmt="%+.2f", fontsize=5.5, padding=2)
    ax.axhline(0, color=PALETTE["blue"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x, ["No product trend", "Improving product"])
    ax.set_ylabel("Elasticity error (estimate $-$ truth)")
    ax.set_ylim(-0.55, 0.15)
    ax.tick_params(axis="x", length=0)
    ax.legend(fontsize=6.5, loc="lower left")

    panel_labels(axes)
    save(fig, "estimates-vs-truth")


if __name__ == "__main__":
    main()
