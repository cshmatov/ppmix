"""
Figures for: "A Mean-Variance Analogue for Mixed Public/Private Equity Portfolios"
All data are simulated / illustrative. Run:  python3 make_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.optimize import minimize, linprog
from scipy.stats import norm
from scipy import sparse
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- palette (validated categorical order) ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 11,
})
rng = np.random.default_rng(7)


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# =====================================================================
# Figure 1: appraisal smoothing and de-smoothing
# =====================================================================
def fig1():
    beta_true, phi = 1.3, 0.60          # true factor loading, AR(1) smoothing parameter
    sF, sE = 0.09, 0.06                 # quarterly factor vol, quarterly idiosyncratic vol
    # --- short sample for the time-series panel
    Tq = 60
    F = rng.normal(0.02, sF, Tq)
    r_true = beta_true * F + rng.normal(0, sE, Tq)
    r_rep = np.zeros(Tq)
    for t in range(Tq):
        r_rep[t] = (1 - phi) * r_true[t] + (phi * r_rep[t - 1] if t > 0 else 0)
    # --- long sample for population-level statistics
    Tl = 40000
    Fl = rng.normal(0.02, sF, Tl)
    rl = beta_true * Fl + rng.normal(0, sE, Tl)
    rr = np.zeros(Tl)
    for t in range(1, Tl):
        rr[t] = (1 - phi) * rl[t] + phi * rr[t - 1]
    # contemporaneous OLS beta on reported returns
    b0 = np.cov(rr, Fl)[0, 1] / Fl.var()
    # Dimson summed-lag beta (lags 0..4)
    K = 4
    X = np.column_stack([Fl[K - k:Tl - k] for k in range(K + 1)])
    y = rr[K:]
    Xc = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(Xc, y, rcond=None)[0][1:]
    b_sum = coef.sum()
    # AR(1) unsmoothing (Geltner):  r_t = (r~_t - phi r~_{t-1})/(1-phi)
    phi_hat = np.corrcoef(rr[1:], rr[:-1])[0, 1]
    r_un = (rr[1:] - phi_hat * rr[:-1]) / (1 - phi_hat)
    ann = np.sqrt(4)
    vols = [rr.std() * ann, r_un.std() * ann, rl.std() * ann]
    cors = [np.corrcoef(rr, Fl)[0, 1], np.corrcoef(r_un, Fl[1:])[0, 1], np.corrcoef(rl, Fl)[0, 1]]
    betas = [b0, b_sum, beta_true]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    q = np.arange(1, Tq + 1)
    ax.plot(q, 100 * r_true, color=BLUE, lw=2, label="Latent economic return  $r_t$")
    ax.plot(q, 100 * r_rep, color=ORANGE, lw=2, label=r"Reported NAV return  $\tilde r_t=(1-\phi)r_t+\phi\tilde r_{t-1}$")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Quarterly return (%)")
    ax.set_title("A. Appraisal smoothing dampens and lags the true return  (φ = 0.6)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.99, 0.03, "simulated:  $r_t = 1.3\\,F_t + \\varepsilon_t$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)

    ax = axes[1]
    labels = ["Naive\n(reported)", "Recovered\n(unsmoothed)", "True"]
    groups = {"Factor loading β": betas, "Annualised vol": vols, "Corr. with factor": cors}
    xg = np.arange(3)
    wbar = 0.26
    cols = [ORANGE, AQUA, BLUE]
    for i, (gname, vals) in enumerate(groups.items()):
        for j, v in enumerate(vals):
            ax.bar(i + (j - 1) * wbar, v, width=wbar * 0.92, color=cols[j],
                   label=labels[j] if i == 0 else None)
            ax.text(i + (j - 1) * wbar, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(xg)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylim(0, 1.95)
    ax.set_title("B. Reported series vs. recovered vs. truth")
    ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.2, "β recovered as the sum of lagged (Dimson) betas, lags 0–4;  vol and correlation recovered by AR(1) unsmoothing",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)
    axes[0].set_title("A. Appraisal smoothing dampens and lags the true return  (φ = 0.6)")
    fig.suptitle("Figure 1 — Private-asset returns are observed through a smoothing filter; the filter must be inverted before any covariance is estimated",
                 fontsize=10.5, color=INK2, x=0.01, ha="left", y=1.02)
    savefig(fig, "fig1_smoothing.png")
    return dict(betas=betas, vols=vols, cors=cors, phi_hat=phi_hat)


# =====================================================================
# Figure 2: efficient frontiers — naive vs de-smoothed vs liquidity-constrained
# =====================================================================
def frontier(mu, Sig, n=60, cap=None):
    N = len(mu)
    cons0 = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * N
    if cap is not None:
        bounds[2] = (0, cap)
    # min-variance and max-return endpoints
    def solve(target):
        cons = cons0 + [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ Sig @ w, np.ones(N) / N, bounds=bounds, constraints=cons, method="SLSQP")
        return res
    mv = minimize(lambda w: w @ Sig @ w, np.ones(N) / N, bounds=bounds, constraints=cons0, method="SLSQP")
    lo = mv.x @ mu
    hi = max(mu) if cap is None else (cap * mu[2] + (1 - cap) * mu[1])
    tgts = np.linspace(lo, hi - 1e-6, n)
    W, S, M = [], [], []
    for t in tgts:
        r = solve(t)
        if r.success:
            W.append(r.x); S.append(np.sqrt(r.x @ Sig @ r.x)); M.append(r.x @ mu)
    return np.array(W), np.array(S), np.array(M)


def fig2():
    names = ["Bonds/cash", "Public equity", "Private equity"]
    mu = np.array([0.03, 0.07, 0.10])
    sd_naive = np.array([0.05, 0.16, 0.17]);   c_naive = 0.50
    sd_true = np.array([0.05, 0.16, 0.26]);    c_true = 0.80
    def cov(sd, c):
        C = np.array([[1, 0.10, 0.05], [0.10, 1, c], [0.05, c, 1]])
        return np.outer(sd, sd) * C
    Sn, St = cov(sd_naive, c_naive), cov(sd_true, c_true)
    gamma = 2.5
    Wn, Sdn, Mn = frontier(mu, Sn)
    Wt, Sdt, Mt = frontier(mu, St)
    Wc, Sdc, Mc = frontier(mu, St, cap=0.25)
    def pick(W, S, M):
        u = M - 0.5 * gamma * S ** 2
        i = int(np.argmax(u)); return W[i], S[i], M[i]
    wn, sn, mn = pick(Wn, Sdn, Mn)
    wt, st, mt = pick(Wt, Sdt, Mt)
    wc, sc, mc = pick(Wc, Sdc, Mc)
    # naive portfolio evaluated with true moments
    sn_true = np.sqrt(wn @ St @ wn)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(100 * Sdn, 100 * Mn, color=ORANGE, lw=2, label="Naive: reported-NAV moments (σ_PE 17%, ρ 0.50)")
    ax.plot(100 * Sdt, 100 * Mt, color=BLUE, lw=2, label="De-smoothed moments (σ_PE 26%, ρ 0.80)")
    ax.plot(100 * Sdc, 100 * Mc, color=AQUA, lw=2, ls="--", label="De-smoothed + liquidity cap  $w_V \\leq 25\\%$")
    for (w, s, m, col, lab, off) in [(wn, sn, mn, ORANGE, "naive optimum", (-150, -34)),
                                     (wt, st, mt, BLUE, "de-smoothed optimum", (14, -30)),
                                     (wc, sc, mc, AQUA, "liquidity-capped optimum", (-190, 2))]:
        ax.scatter(100 * s, 100 * m, s=70, color=col, edgecolor=SURFACE, zorder=5)
        ax.annotate(f"{lab}\nPE {100*w[2]:.0f}% · Pub {100*w[1]:.0f}% · Bond {100*w[0]:.0f}%",
                    (100 * s, 100 * m), xytext=off, textcoords="offset points", fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.6))
    ax.scatter(100 * sn_true, 100 * mn, s=70, facecolor=SURFACE, edgecolor=ORANGE, lw=2, zorder=5)
    ax.annotate("", xy=(100 * sn_true, 100 * mn), xytext=(100 * sn, 100 * mn),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4, ls="--"))
    ax.annotate("the same naive portfolio,\nrisk measured correctly", (100 * sn_true, 100 * mn), xytext=(-46, -44),
                textcoords="offset points", fontsize=8, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8, alpha=0.6))
    for i, (nm, off) in enumerate(zip(names, [(6, -12), (6, -12), (6, 2)])):
        ax.scatter(100 * sd_true[i], 100 * mu[i], marker="s", s=36, color=MUTED, zorder=4)
        ax.annotate(nm if i < 2 else "Private equity (true)", (100 * sd_true[i], 100 * mu[i]), xytext=off, textcoords="offset points", fontsize=8, color=MUTED)
    ax.scatter(100 * sd_naive[2], 100 * mu[2], marker="s", s=36, facecolor=SURFACE, edgecolor=ORANGE, lw=1.5, zorder=4)
    ax.annotate("Private equity\n(as reported)", (100 * sd_naive[2], 100 * mu[2]), xytext=(6, 2), textcoords="offset points", fontsize=8, color=ORANGE)
    ax.set_xlabel("Portfolio volatility (%, annualised)")
    ax.set_ylabel("Expected return (%, annualised)")
    ax.set_title("Figure 3 — Long-only efficient frontiers for a bond / public-equity / private-equity portfolio  (γ = 2.5 optima marked)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(0, 31); ax.set_ylim(2.5, 11)
    savefig(fig, "fig3_frontier.png")
    return dict(naive=(wn, sn, mn, sn_true), true=(wt, st, mt), capped=(wc, sc, mc))


# =====================================================================
# Figure 3: private weights are a state variable (Takahashi–Alexander pacing)
# =====================================================================
def ta_fund(commit=1.0, life=12, bow=2.5, G=0.12, Y=0.02, rc=(0.25, 0.30, 0.35, 0.40, 0.50, 1.0)):
    """Takahashi-Alexander (2002) deterministic fund model, annual steps. Returns arrays over life."""
    T = life
    calls, dists, nav, uncalled = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1)
    uncalled[0] = commit
    for t in range(1, T + 1):
        r = rc[t - 1] if t - 1 < len(rc) else 1.0
        calls[t] = r * uncalled[t - 1]
        uncalled[t] = uncalled[t - 1] - calls[t]
        rd = max(Y, (t / T) ** bow)
        grown = nav[t - 1] * (1 + G)
        dists[t] = rd * grown
        nav[t] = grown + calls[t] - dists[t]
    return calls, dists, nav, uncalled


def fig3():
    calls, dists, nav, unc = ta_fund()
    T = len(nav) - 1
    yrs = np.arange(T + 1)
    net = dists - calls
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.bar(yrs - 0.2, -100 * calls, width=0.4, color=ORANGE, label="Capital calls $C_t$")
    ax.bar(yrs + 0.2, 100 * dists, width=0.4, color=AQUA, label="Distributions $D_t$")
    ax.plot(yrs, 100 * nav, color=BLUE, lw=2.2, marker="o", ms=4, label="NAV $V_t$")
    ax.plot(yrs, 100 * np.cumsum(net), color=VIOLET, lw=1.8, ls="--", label="Cumulative net cash flow (J-curve)")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Fund age (years)"); ax.set_ylabel("% of commitment")
    ax.set_title("A. A single fund: calls, distributions, NAV")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", visible=False)

    # --- a commitment programme: constant commitment rate as % of total wealth
    H = 25
    g_liq, G = 0.06, 0.12
    target = 0.25
    def program(kappa):
        W_liq, funds = 1.0, []   # list of (age, commit)
        w_path, unc_path, ncf_path = [], [], []
        for t in range(H):
            # age existing funds
            nav_tot, unc_tot, cf = 0.0, 0.0, 0.0
            new_funds = []
            for age, cm in funds:
                c, d, n, u = ta_fund(commit=cm)
                if age + 1 <= 12:
                    cf += d[age + 1] - c[age + 1]
                    nav_tot += n[age + 1]; unc_tot += u[age + 1]
                    new_funds.append((age + 1, cm))
            funds = new_funds
            W_liq = W_liq * (1 + g_liq) + cf
            W_tot = W_liq + nav_tot
            # new commitment this year
            funds.append((0, kappa * W_tot))
            unc_tot += kappa * W_tot
            w_path.append(nav_tot / W_tot); unc_path.append(unc_tot / W_tot); ncf_path.append(cf / W_tot)
        return np.array(w_path), np.array(unc_path), np.array(ncf_path)
    # pick kappa so that steady-state weight ~ target
    kappas = np.linspace(0.02, 0.12, 41)
    ss = [program(k)[0][-5:].mean() for k in kappas]
    k_star = kappas[int(np.argmin(np.abs(np.array(ss) - target)))]
    w_path, unc_path, ncf_path = program(k_star)
    ax = axes[1]
    yrs2 = np.arange(1, H + 1)
    ax.plot(yrs2, 100 * w_path, color=BLUE, lw=2.2, label="Private NAV weight $w_{V,t}$")
    ax.plot(yrs2, 100 * unc_path, color=ORANGE, lw=2, label="Uncalled commitments $U_t/W_t$")
    ax.bar(yrs2, 100 * ncf_path, color=AQUA, alpha=0.9, width=0.6, label="Net cash flow to liquid sleeve (%/yr)")
    ax.axhline(100 * target, color=MUTED, lw=1, ls=":", label="Target weight 25%")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Year of the commitment programme"); ax.set_ylabel("% of total wealth")
    ax.set_title(f"B. Reaching a 25% target by committing {100*k_star:.1f}% of wealth each year")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Figure 2 — The private weight is a state variable driven by cash flows; the decision variable is the commitment pace",
                 fontsize=10.5, color=INK2, x=0.01, ha="left")
    savefig(fig, "fig2_pacing.png")
    return dict(kappa=k_star, w_ss=w_path[-5:].mean(), unc_ss=unc_path[-5:].mean())


# =====================================================================
# Figure 4: optimal static option hedge via mean-CVaR LP
# =====================================================================
def bs_price(S0, K, T, r, sig, kind):
    d1 = (np.log(S0 / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def fig4():
    S = 20000
    T, rf = 1.0, 0.04
    # P-measure log-return scenarios for the market factor: normal regime + crash regime (fat left tail)
    crash = rng.random(S) < 0.12
    f = np.where(crash, rng.normal(-0.22, 0.15, S), rng.normal(0.085, 0.12, S))
    ST = np.exp(f)                                       # index level at T (S0 = 1)
    F = ST - 1.0                                         # simple index return
    wP, wV = 0.60, 0.40                                  # public / private sleeves
    bP, bV = 1.00, 1.30                                  # de-smoothed factor loadings
    sP, sV = 0.05, 0.15                                  # residual vols
    rP = np.exp(bP * f + rng.normal(0, sP, S) - 0.5 * sP ** 2) - 1
    rV = np.exp(bV * f + rng.normal(0, sV, S) - 0.5 * sV ** 2) - 1
    W0 = wP * (1 + rP) + wV * (1 + rV)                  # unhedged terminal wealth per $1
    # option universe
    Kp = np.array([0.80, 0.85, 0.90, 0.95, 1.00]); Kc = np.array([1.05, 1.10, 1.15, 1.20])
    def ivol(K): return 0.19 - (0.35 if K <= 1 else 0.12) * np.log(K)   # skewed smile, flatter call wing
    puts = [("put", K) for K in Kp]; calls = [("call", K) for K in Kc]
    opts = puts + calls
    M = len(opts)
    prices = np.array([bs_price(1.0, K, T, rf, ivol(K), kind) for kind, K in opts])
    payoff = np.column_stack([np.maximum(K - ST, 0) if kind == "put" else np.maximum(ST - K, 0) for kind, K in opts])
    hedge_pnl = payoff - prices * np.exp(rf * T)         # premium financed at rf
    alpha = 0.95
    budget = 0.015                                       # net premium outlay cap (1.5% of wealth)
    mean_floor = W0.mean() - 0.010                       # give up at most 1.0% expected return
    # bounds: puts long only; calls written only, up to the public sleeve notional (covered)
    bounds = [(0, 1.5)] * len(puts) + [(-wP, 0.0)] * len(calls)
    # LP variables: [q (M), zeta (1), u (S)]
    nvar = M + 1 + S
    c = np.zeros(nvar); c[M] = 1.0; c[M + 1:] = 1.0 / ((1 - alpha) * S)
    # u_s >= L_s(q) - zeta,  L_s(q) = (1 - W0_s) - hedge_pnl_s @ q
    # => -hedge_pnl_s @ q - zeta - u_s <= -(1 - W0_s)
    A1 = sparse.hstack([sparse.csr_matrix(-hedge_pnl), sparse.csr_matrix(-np.ones((S, 1))), -sparse.identity(S, format="csr")])
    b1 = -(1 - W0)
    # cost budget: prices @ q <= budget
    A2 = sparse.csr_matrix(np.concatenate([prices, np.zeros(1 + S)])[None, :]); b2 = np.array([budget])
    # expected-return floor: -(mean(hedge_pnl) @ q) <= mean(W0) - mean_floor
    A3 = sparse.csr_matrix(np.concatenate([-hedge_pnl.mean(0), np.zeros(1 + S)])[None, :]); b3 = np.array([W0.mean() - mean_floor])
    # covered-call limit: total written call notional <= public sleeve notional  (-sum q_calls <= wP)
    sel = np.array([0.0] * len(puts) + [-1.0] * len(calls))
    A4 = sparse.csr_matrix(np.concatenate([sel, np.zeros(1 + S)])[None, :]); b4 = np.array([wP])
    A = sparse.vstack([A1, A2, A3, A4]).tocsr(); b = np.concatenate([b1, b2, b3, b4])
    lb = bounds + [(-np.inf, np.inf)] + [(0, None)] * S
    res = linprog(c, A_ub=A, b_ub=b, bounds=lb, method="highs")
    assert res.success, res.message
    q = res.x[:M]
    WH = W0 + hedge_pnl @ q
    def stats(W):
        L = 1 - W
        var = np.quantile(L, alpha); cvar = L[L >= var].mean()
        return dict(mean=W.mean() - 1, vol=W.std(), var=var, cvar=cvar, p5=np.quantile(W, 0.05) - 1, maxloss=1 - W.min())
    s0, s1 = stats(W0), stats(WH)
    cost = prices @ q
    # minimum-variance futures hedge for reference
    h_fut = -(wP * bP + wV * bV)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0, 0.55]})
    ax = axes[0]
    bins = np.linspace(0.4, 1.7, 90)
    ax.hist(W0, bins=bins, color=ORANGE, alpha=0.75, label=f"Unhedged  (CVaR₉₅ = {100*s0['cvar']:.1f}%)")
    ax.hist(WH, bins=bins, color=BLUE, alpha=0.7, label=f"Mean–CVaR optimal hedge  (CVaR₉₅ = {100*s1['cvar']:.1f}%)")
    ax.axvline(1 - s0["var"], color=ORANGE, lw=1.2, ls="--"); ax.axvline(1 - s1["var"], color=BLUE, lw=1.2, ls="--")
    ax.set_xlabel("Terminal wealth per $1 (1-year horizon, economic value)")
    ax.set_ylabel("Scenario count"); ax.set_title("A. Hedged vs unhedged terminal wealth (20,000 scenarios)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.68, f"net premium {100*cost:+.2f}% of wealth (cap 1.5%)\nE[return] give-up {100*(s0['mean']-s1['mean']):.2f}% (cap 1.0%)\nvol {100*s0['vol']:.1f}% → {100*s1['vol']:.1f}%\ndashed: 95% VaR thresholds",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax = axes[1]
    Fg = np.linspace(-0.5, 0.5, 400); STg = 1 + Fg; fg = np.log(STg)
    pay_g = np.column_stack([np.maximum(K - STg, 0) if kind == "put" else np.maximum(STg - K, 0) for kind, K in opts]) @ q - cost * np.exp(rf * T)
    unh_g = wP * np.exp(bP * fg) + wV * np.exp(bV * fg) - 1
    ax.plot(100 * Fg, 100 * unh_g, color=ORANGE, lw=2, label="Unhedged portfolio (systematic part)")
    ax.plot(100 * Fg, 100 * (unh_g + pay_g), color=BLUE, lw=2.2, label="Hedged portfolio")
    ax.plot(100 * Fg, 100 * pay_g, color=AQUA, lw=1.8, ls="--", label="Option overlay P&L  $h(F_T)$")
    ax.axhline(0, color=AXIS, lw=0.8); ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Index return $F_T$ (%)"); ax.set_ylabel("P&L (% of initial wealth)")
    ax.set_title("B. Optimal overlay payoff profile")
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[2]
    colsq = [BLUE if kind == "put" else VIOLET for kind, _ in opts]
    ax.bar(np.arange(M), q, color=colsq, width=0.7)
    ax.set_xticks(np.arange(M)); ax.set_xticklabels([f"{'P' if k=='put' else 'C'}{int(round(100*K))}" for k, K in opts], fontsize=8, rotation=60)
    ax.set_title("C. Positions $q_j$"); ax.set_ylabel("index notional per $1 of wealth")
    ax.axhline(0, color=AXIS, lw=0.8); ax.grid(axis="x", visible=False)
    for j, v in enumerate(q):
        if abs(v) > 1e-3:
            ax.text(j, v + (0.02 if v > 0 else -0.02), f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=7.5, color=INK2)
    ax.text(0.5, -0.3, "P = long put, C = written call", transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("Figure 4 — Optimal static option hedge of the mixed portfolio: Rockafellar–Uryasev mean–CVaR linear programme",
                 fontsize=10.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "fig4_hedge.png")
    return dict(q=dict(zip([f"{k}{K:.2f}" for k, K in opts], np.round(q, 3))), prices=np.round(prices, 4),
                stats_unhedged=s0, stats_hedged=s1, cost=cost, h_fut=h_fut, ivol=[round(ivol(K), 3) for _, K in opts])


# =====================================================================
# Figure 5: look-through factor exposures + risk decomposition
# =====================================================================
def fig5():
    factors = ["Market β (de-smoothed)", "Size (small)", "Value", "Momentum", "Quality / profitability", "Growth",
               "Leverage", "Credit spread (HY)", "Illiquidity", "Ind: Technology", "Ind: Healthcare", "Ind: Financials"]
    pub = np.array([1.00, -0.10, 0.10, 0.00, 0.20, 0.00, -0.10, 0.00, 0.00, 0.28, 0.12, 0.14])
    bo = np.array([1.20, 0.80, 0.30, -0.10, -0.30, 0.10, 1.20, 0.60, 0.90, 0.25, 0.20, 0.08])
    vc = np.array([1.40, 1.50, -0.90, 0.20, -1.20, 1.30, -0.60, 0.20, 1.20, 0.65, 0.25, 0.03])
    w = np.array([0.60, 0.25, 0.15])
    tot = w[0] * pub + w[1] * bo + w[2] * vc
    X = np.column_stack([pub, bo, vc, tot])
    cols = ["Public equity\n(60%)", "Private: buyout\n(25%)", "Private: VC\n(15%)", "Total portfolio\n(look-through)"]
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    im = ax.imshow(X, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=8.5)
    ax.set_yticks(range(len(factors))); ax.set_yticklabels(factors, fontsize=8.5)
    ax.grid(False)
    for i in range(len(factors)):
        for j in range(4):
            ax.text(j, i, f"{X[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(X[i, j]) > 0.9 else INK)
    ax.set_title("A. Look-through exposure matrix $X$")
    ax.text(0.0, -0.15, "Style factors in z-score units; market β in beta units; industries as weights.\nPrivate columns come from a fundamentals-based mapping, not from a regression on NAV returns.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="left", va="top")
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5)

    ax = axes[1]
    comp = {"Common factors": [0.90, 0.60, 0.45, 0.78], "Specific (deal / stock)": [0.10, 0.30, 0.45, 0.17], "Valuation noise Ω": [0.00, 0.10, 0.10, 0.05]}
    colsb = [BLUE, ORANGE, AQUA]
    bottom = np.zeros(4)
    for (nm, vals), col in zip(comp.items(), colsb):
        vals = np.array(vals)
        ax.bar(range(4), 100 * vals, bottom=100 * bottom, color=col, width=0.62, label=nm, edgecolor=SURFACE, linewidth=2)
        for j, v in enumerate(vals):
            if v > 0.04:
                ax.text(j, 100 * (bottom[j] + v / 2), f"{100*v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if col != AQUA else INK)
        bottom += vals
    ax.set_xticks(range(4)); ax.set_xticklabels(["Public", "Buyout", "VC", "Total"], fontsize=9)
    ax.set_ylabel("Share of variance (%)"); ax.set_ylim(0, 100)
    ax.set_title("B. Variance decomposition by sleeve\n($x'X\\Sigma_F X'x$  vs  $x'\\Delta x$  vs  $x'\\Omega x$)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Figure 5 — A Barra-style model extended to private sleeves: exposures are mapped, not regressed, and specific risk is larger",
                 fontsize=10.5, color=INK2, x=0.01, ha="left", y=1.04)
    savefig(fig, "fig5_factor_exposures.png")
    return dict(total=np.round(tot, 2))


# =====================================================================
# Figure 6: architecture of the integrated risk / optimisation pipeline
# =====================================================================
def fig6():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
    def box(x, y, w, h, text, fc, ec=None, fs=8.3, tc=INK):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.2)
        ax.add_patch(p); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)
    def arrow(x1, y1, x2, y2, col=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.2))
    tint_b, tint_o, tint_a, tint_v = "#dbe8f9", "#fbe1d6", "#d3f1e5", "#e3dff7"
    # row labels
    for y, lab in [(6.35, "Inputs"), (4.95, "Measurement"), (3.55, "Risk model"), (2.15, "Integration"), (0.75, "Decisions")]:
        ax.text(0.15, y + 0.3, lab, fontsize=9, color=MUTED, rotation=90, va="center", ha="center", fontweight="bold")
    # inputs
    box(0.6, 6.05, 2.8, 0.85, "Public prices, factor returns\n(daily) & option surfaces", tint_b)
    box(3.7, 6.05, 2.8, 0.85, "Private NAVs, capital calls,\ndistributions (quarterly)", tint_o)
    box(6.8, 6.05, 2.8, 0.85, "Deal-level fundamentals: EV, debt,\nsector, geography, stage", tint_o)
    box(9.9, 6.05, 2.8, 0.85, "Secondary-market marks,\nlisted-PE proxies", tint_o)
    # measurement
    box(0.6, 4.65, 2.8, 0.85, "Public exposures $X_P$ (Barra-style\ncross-sectional model)", tint_b)
    box(3.7, 4.65, 2.8, 0.85, "De-smoothing / nowcasting:\nlagged-beta, AR/MA filter, state-space", tint_o)
    box(6.8, 4.65, 2.8, 0.85, "Exposure mapping $X_V$: fundamentals →\nstyle/industry; unlever → relever", tint_o)
    box(9.9, 4.65, 2.8, 0.85, "Cash-flow / pacing model\n(calls, distributions, NAV dynamics)", tint_o)
    # risk model
    box(0.6, 3.25, 4.2, 0.85, "Common factor covariance $\\Sigma_F(h)$ at the decision horizon\n(overlapping/HAC; incl. credit, liquidity, PE-residual factors)", tint_b)
    box(5.1, 3.25, 3.3, 0.85, "Specific risk: $\\Delta_P$ (stocks),\n$\\Delta_V$ (deal dispersion, few names)", tint_a)
    box(8.7, 3.25, 4.0, 0.85, "Valuation noise $\\Omega$, secondary discount $\\delta$,\nstate dynamics of $(V_t, U_t, L_t)$ from the pacing model", tint_a)
    # integration
    box(2.3, 1.85, 8.4, 0.85, "Integrated model:  $\\mu = X\\lambda + \\alpha - \\text{fees} - \\delta$,    $\\Sigma = X\\,\\Sigma_F(h)\\,X' + \\Delta + \\Omega$,   states $(V_t, U_t, L_t)$", tint_v, fs=9)
    # decisions
    box(0.6, 0.45, 3.6, 0.85, "Allocation & pacing optimiser\n(mean–variance with illiquidity dynamics, MPC)", tint_b)
    box(4.7, 0.45, 3.6, 0.85, "Option overlay: mean–CVaR LP\non joint scenarios (Section 2)", tint_b)
    box(8.8, 0.45, 3.9, 0.85, "Monitoring: factor budgets, stress tests,\nnowcast marks, liquidity coverage", tint_b)
    # arrows
    for x in [2.0, 5.1, 8.2, 11.3]:
        arrow(x, 6.05, x, 5.52)
    arrow(2.0, 4.65, 2.3, 4.12); arrow(5.1, 4.65, 3.4, 4.12); arrow(8.2, 4.65, 6.7, 4.12); arrow(8.2, 4.65, 10.2, 4.12); arrow(11.3, 4.65, 11.0, 4.12)
    arrow(5.1, 4.65, 6.3, 4.12)
    for x in [2.7, 6.75, 10.7]:
        arrow(x, 3.25, min(max(x, 2.5), 10.5), 2.72)
    for x in [2.4, 6.5, 10.7]:
        arrow(x, 1.85, x, 1.32)
    arrow(11.3, 4.65, 11.8, 4.12)
    fig.suptitle("Figure 6 — Architecture of an integrated public/private risk and portfolio-construction stack", fontsize=10.5, color=INK2, x=0.01, ha="left")
    savefig(fig, "fig6_architecture.png")


if __name__ == "__main__":
    out = {}
    out["fig1"] = fig1()
    out["fig2"] = fig2()
    out["fig3"] = fig3()
    out["fig4"] = fig4()
    out["fig5"] = fig5()
    fig6()
    import json
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    with open(os.path.join(OUT, "figure_stats.json"), "w") as f:
        json.dump(conv(out), f, indent=1)
    print(json.dumps(conv(out), indent=1))
