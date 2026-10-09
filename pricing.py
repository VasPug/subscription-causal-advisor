"""Pricing decision: raise the monthly price 10 -> 12?

History: 36 monthly signup cohorts. Conversion depends on price, season, and a
latent quality trend q_t. If the business ever changed price, it did so
endogenously -- right when quality was trending up -- which confounds any
before/after comparison. New-cohort pricing only (no grandfathering issues).
"""
import numpy as np

from methods import logit_fit

P0, P1_HIST, P1 = 10.0, 11.0, 12.0
T_HIST = 36
MONTHS_FWD = 12
SIGNUPS_M = 400


def _season(t):
    return 0.3 * np.sin(2 * np.pi * (np.asarray(t) % 12) / 12.0)


def _conv_p(price, q, s, b_p):
    z = -2.5 - b_p * (price - P0) + 0.5 * q + 0.3 * s
    return 1.0 / (1.0 + np.exp(-z))


def _retained_value(price, rng, n_payers, months_left):
    """12-mo revenue per converting user at a given price (churn rises with price)."""
    p_ch = 1.0 / (1.0 + np.exp(-(-2.2 + 0.08 * (price - P0))))
    alive = np.ones(n_payers, bool)
    rev = np.zeros(n_payers)
    for _ in range(months_left):
        rev += np.where(alive, price, 0.0)
        alive = alive & (rng.random(n_payers) >= p_ch)
    return rev


def make_history(cfg, rng):
    """Returns user-level signup records over 36 months."""
    drift = cfg["q_drift"]
    q = np.cumsum(rng.normal(drift, 0.05, T_HIST))
    if cfg["has_variation"]:
        # endogenous timing: price went up right where q had its best 3-mo upswing
        up = np.convolve(np.diff(q, prepend=q[0]), np.ones(3) / 3, mode="same")
        t0 = int(np.clip(np.argmax(up[6:30]) + 6, 6, 30))
        price_t = np.where(np.arange(T_HIST) < t0, P0, P1_HIST)
    else:
        price_t = np.full(T_HIST, P0)
    rows_t, rows_price, rows_conv, rows_season = [], [], [], []
    for t in range(T_HIST):
        n = rng.poisson(SIGNUPS_M)
        s = _season(t)
        conv = rng.random(n) < _conv_p(price_t[t], q[t], s, cfg["b_p"])
        rows_t.append(np.full(n, t))
        rows_price.append(np.full(n, price_t[t]))
        rows_season.append(np.full(n, s))
        rows_conv.append(conv)
    return dict(t=np.concatenate(rows_t), price=np.concatenate(rows_price),
                season=np.concatenate(rows_season),
                conv=np.concatenate(rows_conv).astype(float),
                q_now=q[-1])


def oracle_delta(cfg, q_now, seed):
    """True 12-mo profit delta per signup of moving new cohorts to P1."""
    rng = np.random.default_rng(seed)
    out = {}
    for price in (P0, P1):
        n = 200000
        conv = _conv_p(price, q_now, 0.0, cfg["b_p"])  # season averages out
        payers = int(n * conv)
        rev = _retained_value(price, rng, payers, MONTHS_FWD).sum()
        out[price] = rev / n
    return out[P1] - out[P0]


def profit_per_signup(b_hat, q_level, conv_obs, seed):
    """Shared business template: profit at each price given an elasticity estimate.

    Uses the observed current conversion rate as the anchor so no method needs q.
    """
    rng = np.random.default_rng(seed)
    base_logit = np.log(conv_obs / (1 - conv_obs))
    out = {}
    for price in (P0, P1):
        z = base_logit - b_hat * (price - P0)
        conv = 1.0 / (1.0 + np.exp(-z))
        payers = max(int(100000 * conv), 1)
        rev = _retained_value(price, rng, payers, MONTHS_FWD).sum()
        out[price] = rev / 100000
    return out[P1] - out[P0]


def recent_conv(hist, months=6):
    sel = hist["t"] >= T_HIST - months
    return float(np.clip(hist["conv"][sel].mean(), 1e-3, 1 - 1e-3))


def naive_pricing(hist, seed):
    """Before/after contrast around the past change; benchmark elasticity 0 if none."""
    changed = hist["price"] > P0
    if changed.any():
        after = hist["conv"][changed].mean()
        before = hist["conv"][~changed].mean()
        la = np.log(np.clip(after, 1e-3, 1)) - np.log(np.clip(before, 1e-3, 1))
        b_hat = -la / (P1_HIST - P0)
    else:
        b_hat = 0.0  # "competitor benchmark says demand is inelastic"
    d = profit_per_signup(b_hat, None, recent_conv(hist), seed)
    return dict(b_hat=b_hat, action="raise" if d > 0 else "hold")


def advisor_pricing(hist, seed):
    changed = hist["price"] > P0
    if not changed.any():
        return dict(b_hat=float("nan"), action="experiment")  # no identification
    X = np.column_stack([hist["price"] - P0, hist["t"] / 12.0, hist["season"]])
    w, cov = logit_fit(X, hist["conv"], with_cov=True)
    b_hat = -w[1]
    se = float(np.sqrt(cov[1, 1]))
    conv_now = recent_conv(hist)
    d_lo = profit_per_signup(b_hat + 1.96 * se, None, conv_now, seed)
    d_hi = profit_per_signup(max(b_hat - 1.96 * se, 0.0), None, conv_now, seed)
    if (d_lo > 0) != (d_hi > 0):
        return dict(b_hat=b_hat, action="experiment")
    d = profit_per_signup(b_hat, None, conv_now, seed)
    return dict(b_hat=b_hat, action="raise" if d > 0 else "hold")


def ab_pricing(cfg, hist, seed):
    """2-month price test on new signups: clean conversion read, truncated value read."""
    rng = np.random.default_rng(seed + 3)
    n = SIGNUPS_M * 2  # two months of signups in the test
    arm = rng.random(n) < 0.5
    q = hist["q_now"]
    conv0 = rng.random((~arm).sum()) < _conv_p(P0, q, 0.0, cfg["b_p"])
    conv1 = rng.random(arm.sum()) < _conv_p(P1, q, 0.0, cfg["b_p"])
    # 2-month revenue per signup, extrapolated x6 (misses churn compounding)
    r0 = _retained_value(P0, rng, int(conv0.sum()), 2).sum() / max((~arm).sum(), 1)
    r1 = _retained_value(P1, rng, int(conv1.sum()), 2).sum() / max(arm.sum(), 1)
    d_hat = 6.0 * (r1 - r0)  # ponytail: naive x6 extrapolation, like the gate A/B
    true_d = cfg["_delta_true"]
    # exposure: half the test signups ride the worse price; delay: everyone else
    # waits 2 months for the better one. Units: $ per annual-cohort signup.
    exposure = 0.5 * (n / (SIGNUPS_M * 12)) * abs(true_d)
    delay = max(true_d, 0.0) * (2.0 / 12.0)
    return dict(b_hat=float("nan"),
                action="raise" if d_hat > 0 else "hold",
                cost=exposure + delay)
