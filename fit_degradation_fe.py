"""
Phase 1, step 5: tyre degradation with two-way fixed effects.

Why: in the earlier models, compound choice is tangled up with *when* in the race that
compound is used (softs early, hards late), and the fuel/track-evolution trend was forced
to be a straight line. Here every race-lap gets its own effect, which absorbs fuel burn,
track evolution, safety-car laps and weather for all drivers at once:

    lap_time = driver_race_effect + race_lap_effect
               + pace_c + a_c * TyreAge + b_c * TyreAge^2

Tyre effects are then learned only from comparing drivers on the SAME lap who are on
different tyres/ages. A race-level bootstrap gives honest confidence intervals.

Usage:   python fit_degradation_fe.py        (needs data/*_ALL_laps.csv)
         python fit_degradation_fe.py 2025   (hold out 2025: for a clean out-of-sample backtest)
Output:  models/tyre_model_fe.json, tyre_degradation_fe.png, printed summary
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fit_degradation import COMPOUNDS, load  # noqa: E402

FEATURES = [f"{c}_{t}" for c in COMPOUNDS for t in ("age", "age2")] + ["MEDIUM_off", "HARD_off"]
N_BOOT = 100
HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else None  # e.g. 2025: fit WITHOUT that season
SUFFIX = f"_holdout{HOLDOUT}" if HOLDOUT else ""
DIRTY_AIR_S = 1.5  # drop laps run closer than this to the car ahead


def build_X(df: pd.DataFrame) -> np.ndarray:
    age = df["TyreAge"].to_numpy(float)
    cols = {}
    for c in COMPOUNDS:
        m = (df["Compound"] == c).to_numpy(float)
        cols[f"{c}_age"], cols[f"{c}_age2"] = m * age, m * age ** 2
    for c in COMPOUNDS[1:]:
        cols[f"{c}_off"] = (df["Compound"] == c).to_numpy(float)
    return np.column_stack([cols[k] for k in FEATURES])


def demean2(M, g1, g2, n1, n2, iters=200, tol=1e-7):
    """Remove group means for two sets of groups at once (alternating projections)."""
    M = M.copy()
    c1 = np.maximum(np.bincount(g1, minlength=n1), 1)
    c2 = np.maximum(np.bincount(g2, minlength=n2), 1)
    for _ in range(iters):
        biggest = 0.0
        for g, c in ((g1, c1), (g2, c2)):
            for j in range(M.shape[1]):
                m = np.bincount(g, weights=M[:, j], minlength=len(c)) / c
                M[:, j] -= m[g]
                biggest = max(biggest, float(np.abs(m).max()))
        if biggest < tol:
            break
    return M


def prepare():
    df = load()
    if HOLDOUT:
        before = len(df)
        df = df[~df["Race"].astype(str).str.startswith(f"{HOLDOUT}_")].copy()
        print(f"Hold-out mode: fitting WITHOUT {HOLDOUT} ({before - len(df)} laps left out)")
    if "GapAhead" in df.columns and df["GapAhead"].notna().any():
        dirty = df["GapAhead"] < DIRTY_AIR_S
        print(f"Dropping {int(dirty.sum())} laps run within {DIRTY_AIR_S}s of the car ahead "
              f"({100 * dirty.mean():.1f}% of clean laps)")
        df = df[~dirty].copy()
    else:
        print("No GapAhead column found: re-download with the updated load_openf1.py to filter dirty-air laps.")
    df["rl"] = df["Race"].astype(str) + "|" + df["LapNumber"].astype(int).astype(str)
    # Keep race-laps with enough drivers to compare, and driver-races with enough laps.
    for _ in range(2):
        df = df[df.groupby("rl")["LapTimeSec"].transform("size") >= 6]
        df = df[df.groupby("grp")["LapTimeSec"].transform("size") >= 10]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = prepare()
    Y = df["LapTimeSec"].to_numpy(float)
    X = build_X(df)
    g1 = pd.factorize(df["grp"])[0]
    g2 = pd.factorize(df["rl"])[0]
    n1, n2 = g1.max() + 1, g2.max() + 1
    race_rows = {r: np.flatnonzero((df["Race"] == r).to_numpy()) for r in df["Race"].unique()}
    races = np.array(list(race_rows))
    print(f"{len(df)} clean laps, {len(races)} races, {n1} driver-races, {n2} race-laps")

    def fit_rows(rows):
        M = np.column_stack([Y[rows], X[rows]])
        Md = demean2(M, g1[rows], g2[rows], n1, n2)
        beta = np.linalg.lstsq(Md[:, 1:], Md[:, 0], rcond=None)[0]
        return beta, Md

    # Held-out check on ~20% of races (their own race-lap effects are re-estimated, so this
    # measures how much of the remaining variation the tyre terms explain on unseen races).
    rng = np.random.default_rng(0)
    test_races = set(rng.choice(races, size=max(1, len(races) // 5), replace=False))
    tr = np.flatnonzero(~df["Race"].isin(test_races).to_numpy())
    te = np.flatnonzero(df["Race"].isin(test_races).to_numpy())
    beta_tr, _ = fit_rows(tr)
    _, Md_te = fit_rows(te)
    yd, Xd = Md_te[:, 0], Md_te[:, 1:]
    base = float(np.sqrt(np.mean(yd ** 2)))
    model = float(np.sqrt(np.mean((yd - Xd @ beta_tr) ** 2)))
    print(f"\nHeld-out races ({len(test_races)}), after removing driver and race-lap effects:")
    print(f"  RMSE without tyre terms {base:.3f}s, with tyre terms {model:.3f}s "
          f"({100 * (1 - model / base):.1f}% lower)")

    # Final fit and race-level bootstrap.
    all_rows = np.arange(len(df))
    beta, _ = fit_rows(all_rows)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(races, size=len(races), replace=True)
        rows = np.concatenate([race_rows[r] for r in pick])
        boots.append(fit_rows(rows)[0])
    boots = np.array(boots)

    idx = {k: i for i, k in enumerate(FEATURES)}
    p95 = {c: float(df.loc[df["Compound"] == c, "TyreAge"].quantile(0.95)) for c in COMPOUNDS}

    def loss(b, c, age):
        return b[idx[f"{c}_age"]] * age + b[idx[f"{c}_age2"]] * age ** 2

    def ci(vals):
        lo, hi = np.percentile(vals, [5, 95])
        return f"[{lo:+.2f}, {hi:+.2f}]"

    print("\nFresh-tyre pace vs a fresh soft (s/lap, 90% interval):")
    for c in COMPOUNDS[1:]:
        j = idx[f"{c}_off"]
        print(f"  {c:<7}{beta[j]:+.2f}  {ci(boots[:, j])}")

    ages = [5, 10, 15, 20, 25, 30]
    print("\nTime lost vs a fresh tyre of the same compound (s/lap):")
    print("age".ljust(8) + "".join(f"{a:>8}" for a in ages) + "   at age 15, 90% interval")
    for c in COMPOUNDS:
        cells = [f"{loss(beta, c, a):8.2f}" if a <= p95[c] else "       -" for a in ages]
        print(c.ljust(8) + "".join(cells) + f"   {ci([loss(b, c, 15) for b in boots])}")
    print("('-' = beyond the tyre ages seen in the data)")

    # Average lap-time trend across a race (fuel burn + track evolution), from the race-lap effects.
    r = Y - X @ beta
    c1 = np.maximum(np.bincount(g1, minlength=n1), 1)
    c2 = np.maximum(np.bincount(g2, minlength=n2), 1)
    a = np.zeros(len(df))
    for _ in range(200):
        d = (np.bincount(g1, weights=r - a, minlength=n1) / c1)[g1]
        a = (np.bincount(g2, weights=r - d, minlength=n2) / c2)[g2]
    lapn = df["LapNumber"].to_numpy(float)
    rc = pd.factorize(df["Race"])[0]
    cnt = np.bincount(rc)
    lap_d = lapn - (np.bincount(rc, weights=lapn) / cnt)[rc]
    a_d = a - (np.bincount(rc, weights=a) / cnt)[rc]
    trend = float((lap_d * a_d).sum() / (lap_d ** 2).sum())
    print(f"\nAverage lap-time trend over a race (fuel burn + track evolution): {trend:+.4f} s per lap")

    Path("models").mkdir(exist_ok=True)
    Path(f"models/tyre_model_fe{SUFFIX}.json").write_text(json.dumps({
        "fuel_s_per_lap": trend,
        "pace_offset_vs_soft": {"SOFT": 0.0, "MEDIUM": float(beta[idx["MEDIUM_off"]]),
                                "HARD": float(beta[idx["HARD_off"]])},
        "compounds": {c: {"lin": float(beta[idx[f"{c}_age"]]), "quad": float(beta[idx[f"{c}_age2"]]),
                          "max_age_p95": p95[c]} for c in COMPOUNDS},
        "n_laps": int(len(df)), "n_races": int(len(races)), "n_bootstrap": N_BOOT,
        "features": FEATURES, "boot_betas": np.round(boots, 6).tolist(),
        "validation": {"rmse_without_tyre_terms": base, "rmse_with_tyre_terms": model},
    }, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "tab:gray"}
    for c in COMPOUNDS:
        x = np.linspace(0, p95[c], 50)
        curves = np.array([loss(b, c, x) for b in boots])
        ax.plot(x, loss(beta, c, x), color=colors[c], lw=2, label=c)
        ax.fill_between(x, np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0),
                        color=colors[c], alpha=0.18)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Time lost vs fresh tyre (s/lap)")
    ax.set_title("Tyre degradation, two-way fixed effects (90% bootstrap band)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"tyre_degradation_fe{SUFFIX}.png", dpi=150)
    print(f"\nSaved models/tyre_model_fe{SUFFIX}.json and tyre_degradation_fe{SUFFIX}.png")
