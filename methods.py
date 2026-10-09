"""Three decision-makers: naive contrast, causal advisor (AIPW-ATT + abstain), A/B test."""
import numpy as np

from sim import decide_from_tau


def logit_fit(X, y, l2=1e-3, iters=60):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + l2 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def ols_fit(X, y, l2=1e-6):
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(Xb.T @ Xb + l2 * np.eye(Xb.shape[1]), Xb.T @ y)


def _pred_lin(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def att_tau(X, D, Y, idx=None, drop_col=None):
    """Doubly-robust ATT of adoption on profit (effect among the treated)."""
    if idx is not None:
        X, D, Y = X[idx], D[idx], Y[idx]
    if drop_col is not None:
        X = np.delete(X, drop_col, axis=1)
    e = 1.0 / (1.0 + np.exp(-np.clip(_pred_lin(logit_fit(X, D), X), -30, 30)))
    e = np.clip(e, 0.02, 0.98)
    c = D < 0.5
    m0 = _pred_lin(ols_fit(X[c], Y[c]), X)
    w = e / (1 - e)
    psi = D * (Y - m0) - (1 - D) * w * (Y - m0)
    return psi.mean() / D.mean()


def naive(data):
    t = data["D"] > 0.5
    tau_hat = data["Y"][t].mean() - data["Y"][~t].mean()
    gate, _ = decide_from_tau(tau_hat, data)
    return dict(tau_hat=tau_hat, action="gate" if gate else "keep")


def advisor(data, rng, n_boot=80):
    X, D, Y = data["X"], data["D"], data["Y"]
    tau_hat = att_tau(X, D, Y)
    n = len(Y)
    boots = np.array([
        att_tau(X, D, Y, idx=rng.integers(0, n, n)) for _ in range(n_boot)
    ])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # sensitivity: how much does tau move if we lose the most load-bearing
    # covariate? treat that as the plausible scale of leftover hidden bias.
    shifts = [abs(att_tau(X, D, Y, drop_col=j) - tau_hat) for j in range(X.shape[1])]
    sens = max(shifts)
    lo_s, hi_s = lo - sens, hi + sens
    gate_lo, _ = decide_from_tau(lo_s, data)
    gate_hi, _ = decide_from_tau(hi_s, data)
    if gate_lo != gate_hi:  # decision flips within uncertainty -> ask for an experiment
        return dict(tau_hat=tau_hat, ci=(lo, hi), sens=sens, action="experiment")
    gate, _ = decide_from_tau(tau_hat, data)
    return dict(tau_hat=tau_hat, ci=(lo, hi), sens=sens,
                action="gate" if gate else "keep")


def ab_test(pop, orc, seed, frac=0.1):
    """Randomized gate on a cohort for 3 months, extrapolated to the horizon.

    Unbiased short-run, but pays exposure + delay costs and extrapolates 3->12mo.
    """
    from sim import simulate_profit
    rng = np.random.default_rng(seed + 7)
    n = len(pop["a"])
    test = rng.random(n) < frac
    arm = rng.random(n) < 0.5
    rng_a = np.random.default_rng(seed + 11)
    p_g, _ = simulate_profit(pop, gated=True, months=3, rng=rng_a)
    rng_b = np.random.default_rng(seed + 11)
    p_u, _ = simulate_profit(pop, gated=False, months=3, rng=rng_b)
    tg, tu = test & arm, test & ~arm
    delta_hat = 4.0 * (p_g[tg].mean() - p_u[tu].mean())  # ponytail: x4 extrapolation, no dynamics
    exposure = -(frac * 0.5) * (orc["delta_short"] if orc["delta_short"] < 0 else 0.0)
    delay = (1 - frac) * max(orc["delta_true"], 0.0) * (3.0 / 12.0)
    return dict(delta_hat=delta_hat,
                action="gate" if delta_hat > 0 else "keep",
                cost=exposure + delay)
