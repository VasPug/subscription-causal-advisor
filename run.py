"""Packaging decision: naive vs causal advisor vs A/B test, scored against the oracle.

Prints per-config and summary metrics to stdout; writes results.csv.
"""
import csv
import json
import sys
import time

import numpy as np

import sim
from methods import naive, advisor, ab_test

CFG = dict(
    experiment="pricing",
    n_users=12000,
    replicates=30,
    tau_x_grid=[1.0, 8.0],          # feature value: low (gate likely pays) vs high
    conf_grid=[0.0, 0.6, 1.2, 1.8], # hidden-confounding strength
    b_p_grid=[0.02, 0.45],          # price elasticity: inelastic (raise pays) vs elastic
    variation_grid=[True, False],   # did price ever change historically?
    q_drift_grid=[0.0, 0.04],       # latent quality trend (confounds before/after)
)
if "--smoke" in sys.argv:
    CFG.update(n_users=3000, replicates=3)


def value(action_gate, delta_true):
    return delta_true if action_gate else 0.0


def run_pricing():
    import pricing as pr
    rows = []
    for b_p in CFG["b_p_grid"]:
        for has_var in CFG["variation_grid"]:
            for drift in CFG["q_drift_grid"]:
                for rep in range(CFG["replicates"]):
                    seed = hash((b_p, has_var, drift, rep)) % 2**31
                    rng = np.random.default_rng(seed)
                    cfg = dict(b_p=b_p, has_variation=has_var, q_drift=drift)
                    hist = pr.make_history(cfg, rng)
                    delta_true = pr.oracle_delta(cfg, hist["q_now"], seed + 1)
                    cfg["_delta_true"] = delta_true
                    best = max(delta_true, 0.0)

                    res_n = pr.naive_pricing(hist, seed + 2)
                    res_a = pr.advisor_pricing(hist, seed + 2)
                    res_b = pr.ab_pricing(cfg, hist, seed)

                    for name, res in [("naive", res_n), ("advisor", res_a), ("ab", res_b)]:
                        cost = res.get("cost", 0.0)
                        action = res["action"]
                        abstained = action == "experiment"
                        if abstained:
                            action, cost = res_b["action"], res_b["cost"]
                        chosen = delta_true if action == "raise" else 0.0
                        rows.append(dict(
                            b_p=b_p, has_var=int(has_var), drift=drift, rep=rep,
                            method=name, action=action, abstained=int(abstained),
                            correct=int((action == "raise") == (delta_true > 0)),
                            regret=best - (chosen - cost),
                            b_hat=res.get("b_hat", float("nan")),
                            delta_true=delta_true,
                        ))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        return dict(
            acc=sum(x["correct"] for x in r) / len(r),
            regret=1000 * sum(x["regret"] for x in r) / len(r),
            abstain=sum(x["abstained"] for x in r) / len(r),
            bias=float(np.mean([x["b_hat"] - x["b_p"] for x in r
                                if np.isfinite(x["b_hat"])] or [float("nan")])),
        )

    print(f"\n== pricing experiment: {CFG['replicates']} reps ==")
    print(f"{'b_p':>5} {'var':>4} {'drift':>5} | {'method':>8} {'acc':>5} "
          f"{'regret$/1k':>10} {'b-bias':>7} {'abstain':>7}")
    for b_p in CFG["b_p_grid"]:
        for has_var in CFG["variation_grid"]:
            for drift in CFG["q_drift_grid"]:
                dt = np.mean([x["delta_true"] for x in rows
                              if x["b_p"] == b_p and x["has_var"] == int(has_var)
                              and x["drift"] == drift and x["method"] == "naive"])
                print(f"-- true raise delta: {dt:+.3f} $/signup/yr --")
                for m in ["naive", "advisor", "ab"]:
                    s = agg(lambda x, b=b_p, v=int(has_var), d=drift, mm=m:
                            x["b_p"] == b and x["has_var"] == v
                            and x["drift"] == d and x["method"] == mm)
                    print(f"{b_p:>5} {has_var!s:>4} {drift:>5} | {m:>8} {s['acc']:>5.2f} "
                          f"{s['regret']:>10.1f} {s['bias']:>7.2f} {s['abstain']:>7.2f}")
    print("\n== overall ==")
    summary = {}
    for m in ["naive", "advisor", "ab"]:
        s = agg(lambda x, mm=m: x["method"] == mm)
        summary[m] = s
        print(f"{m:>8}: accuracy={s['acc']:.3f} regret=${s['regret']:.1f}/1k-signups/yr "
              f"b-bias={s['bias']:+.3f} abstain-rate={s['abstain']:.2f}")
    print("RESULT " + json.dumps(summary))


def main():
    t0 = time.time()
    print("CONFIG " + json.dumps(CFG))
    if CFG["experiment"] == "pricing":
        run_pricing()
        print(f"done in {time.time() - t0:.0f}s")
        return
    rows = []
    for tau_x in CFG["tau_x_grid"]:
        for conf in CFG["conf_grid"]:
            for rep in range(CFG["replicates"]):
                seed = hash((tau_x, conf, rep)) % 2**31
                rng = np.random.default_rng(seed)
                pop = sim.draw_population(CFG["n_users"], dict(tau_x=tau_x, conf=conf), rng)
                orc = sim.oracle(pop, seed + 1)
                data = sim.observational_dataset(pop, seed + 2)
                best = value(orc["delta_true"] > 0, orc["delta_true"])

                res_n = naive(data)
                res_a = advisor(data, rng)
                res_b = ab_test(pop, orc, seed)

                for name, res in [("naive", res_n), ("advisor", res_a), ("ab", res_b)]:
                    cost = res.get("cost", 0.0)
                    action = res["action"]
                    abstained = action == "experiment"
                    if abstained:  # advisor defers to a designed experiment
                        action, cost = res_b["action"], res_b["cost"]
                    reg = best - (value(action == "gate", orc["delta_true"]) - cost)
                    rows.append(dict(
                        tau_x=tau_x, conf=conf, rep=rep, method=name,
                        action=action, abstained=int(abstained),
                        correct=int((action == "gate") == (orc["delta_true"] > 0)),
                        regret=reg,
                        tau_hat=res.get("tau_hat", float("nan")),
                        tau_true=orc["tau_true"],
                        delta_true=orc["delta_true"],
                    ))

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    def agg(sel):
        r = [x for x in rows if sel(x)]
        n = len(r)
        return dict(
            acc=sum(x["correct"] for x in r) / n,
            regret=1000 * sum(x["regret"] for x in r) / n,  # $ per 1000 users/yr
            abstain=sum(x["abstained"] for x in r) / n,
            bias=float(np.mean([x["tau_hat"] - x["tau_true"] for x in r
                                if np.isfinite(x["tau_hat"])] or [float("nan")])),
        )

    print(f"\n== packaging experiment: {CFG['n_users']} users x {CFG['replicates']} reps ==")
    hdr = f"{'tau_x':>6} {'conf':>5} | {'method':>8} {'acc':>5} {'regret$/1k':>10} {'bias':>7} {'abstain':>7}"
    print(hdr)
    for tau_x in CFG["tau_x_grid"]:
        for conf in CFG["conf_grid"]:
            dt = np.mean([x["delta_true"] for x in rows
                          if x["tau_x"] == tau_x and x["conf"] == conf and x["method"] == "naive"])
            print(f"-- true gate delta: {dt:+.3f} $/user/yr --")
            for m in ["naive", "advisor", "ab"]:
                s = agg(lambda x, t=tau_x, c=conf, mm=m:
                        x["tau_x"] == t and x["conf"] == c and x["method"] == mm)
                print(f"{tau_x:>6} {conf:>5} | {m:>8} {s['acc']:>5.2f} {s['regret']:>10.1f} "
                      f"{s['bias']:>7.2f} {s['abstain']:>7.2f}")

    print("\n== overall ==")
    summary = {}
    for m in ["naive", "advisor", "ab"]:
        s = agg(lambda x, mm=m: x["method"] == mm)
        summary[m] = s
        print(f"{m:>8}: accuracy={s['acc']:.3f} regret=${s['regret']:.1f}/1k-users/yr "
              f"tau-bias={s['bias']:+.2f} abstain-rate={s['abstain']:.2f}")
    # confounded-only slice (conf>0): where the thesis lives
    for m in ["naive", "advisor", "ab"]:
        s = agg(lambda x, mm=m: x["method"] == mm and x["conf"] > 0)
        print(f"confounded-only {m:>8}: accuracy={s['acc']:.3f} regret=${s['regret']:.1f}/1k "
              f"bias={s['bias']:+.2f}")
    print("RESULT " + json.dumps({m: summary[m] for m in summary}))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
