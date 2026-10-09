"""Synthetic subscription business with known ground truth.

World: free users may adopt feature X (endogenously -> confounded).
Decision under study: gate X behind the paid tier.
Oracle simulates both policies with common random numbers.

Confounding has two layers:
- engagement `a` drives adoption AND outcomes, observed only through noisy proxies;
- hidden taste `h` drives outcomes always, and adoption with strength cfg["conf"]
  (the unobserved-confounding dial; h never appears in any covariate).
"""
import numpy as np

PRICE_M = 10.0      # $/month premium
FREE_MARGIN_M = 2.0 # $/month margin on a free user (ads etc.)
HORIZON = 12        # months scored


def draw_population(n, cfg, rng):
    a = rng.gamma(2.0, 1.0, n)                      # latent engagement
    h = rng.normal(0, 1, n)                         # hidden taste (unobserved)
    wtp = np.clip(2.0 + 3.0 * a + 2.0 * rng.normal(0, 1, n), 0, None)  # $/mo
    p_adopt = _sig(-1.0 + 0.9 * a + cfg["conf"] * h)
    adopter = rng.random(n) < p_adopt
    xval = cfg["tau_x"] * (0.3 + 0.7 * a / 2.0)     # heterogeneous feature value
    obs = np.column_stack([
        a + rng.normal(0, 0.4, n),
        a + rng.normal(0, 0.8, n),
        rng.normal(0, 1, n),
    ])
    wtp_proxy = wtp + rng.normal(0, 1.0, n)
    return dict(a=a, h=h, wtp=wtp, adopter=adopter, xval=xval, obs=obs,
                wtp_proxy=wtp_proxy)


def simulate_profit(pop, gated, months, rng, force_access=None):
    """Month loop: upgrade, churn, revenue. force_access overrides free-tier access
    (oracle uses it to define the access ATT among adopters)."""
    n = len(pop["a"])
    alive = np.ones(n, bool)
    paid = np.zeros(n, bool)
    profit = np.zeros(n)
    u = rng.random((months, n))
    v = rng.random((months, n))
    for t in range(months):
        if force_access is None:
            access = paid | (pop["adopter"] & (not gated))
        else:
            access = paid | (pop["adopter"] & (force_access > 0))
        got_val = access & pop["adopter"]
        val = 0.8 * pop["a"] + np.where(got_val, 1.1 * pop["xval"], 0.0)
        unlock = (pop["adopter"] & ~paid & gated) if force_access is None \
            else np.zeros(n, bool)
        unlock_push = np.where(
            unlock, 0.35 * np.clip(pop["wtp"] + 0.15 * pop["xval"] - PRICE_M, 0, None), 0.0)
        p_up = _sig(-7.2 + 0.28 * pop["wtp"] + 0.1 * pop["a"] + 0.3 * pop["h"]
                    + unlock_push)
        newly = alive & ~paid & (u[t] < p_up)
        paid = paid | newly
        # gated-out adopters churn extra: frustration at losing a feature they use
        p_ch = _sig(-1.9 - 0.45 * val - 0.35 * pop["h"] + np.where(paid, -0.5, 0.0)
                    + np.where(unlock, 0.16 * pop["xval"], 0.0))
        churned = alive & (v[t] < p_ch)
        profit += np.where(alive, np.where(paid, PRICE_M, FREE_MARGIN_M), 0.0)
        alive = alive & ~churned
    return profit, paid


def oracle(pop, seed):
    """True per-user annual profit under each policy + access ATT among adopters."""
    def run(gated, months, force_access=None):
        rng = np.random.default_rng(seed)  # common random numbers across policies
        return simulate_profit(pop, gated, months, rng, force_access)

    p_ungated, _ = run(False, HORIZON)
    p_gated, _ = run(True, HORIZON)
    p_acc1, _ = run(False, HORIZON, force_access=np.ones(len(pop["a"])))
    p_acc0, _ = run(False, HORIZON, force_access=np.zeros(len(pop["a"])))
    ad = pop["adopter"]
    tau_true = (p_acc1[ad] - p_acc0[ad]).mean() if ad.any() else 0.0
    s_un, _ = run(False, 3)
    s_g, _ = run(True, 3)
    return dict(delta_true=(p_gated - p_ungated).mean(),
                tau_true=tau_true,
                delta_short=(s_g - s_un).mean())


def observational_dataset(pop, seed):
    """Status-quo (ungated) world: what the business has in its warehouse."""
    rng = np.random.default_rng(seed)
    profit, paid = simulate_profit(pop, gated=False, months=HORIZON, rng=rng)
    return dict(X=pop["obs"], D=pop["adopter"].astype(float), Y=profit,
                wtp_proxy=pop["wtp_proxy"], paid=paid)


def decide_from_tau(tau_hat, data):
    """Shared business template mapping an access-value estimate to gate/no-gate.

    ponytail: stylized linear profit model shared by every method; oracle-sign
    consistency at the true tau is asserted in test_sanity.py, not derived.
    """
    d = data["D"] > 0.5
    payshare = (data["wtp_proxy"][d] >= PRICE_M).mean() if d.any() else 0.0
    pull_rev = payshare * PRICE_M * HORIZON * 0.08   # few payer-capable users unlock
    loss = (1.0 - payshare) * max(tau_hat, 0.0)
    delta_est = d.mean() * (pull_rev - loss)
    return (delta_est > 0.0), delta_est


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
