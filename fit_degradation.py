"""
Phase 1, step 3: fit a tyre-degradation model on every race in data/.

Model (per driver per race, so car speed and driver pace cancel out):
    lap_time = driver_race_baseline
               + fuel * LapNumber                         (cars get lighter and faster)
               + pace_c                                   (fresh-tyre pace gap: MEDIUM/HARD vs SOFT)
               + a_c * TyreAge + b_c * TyreAge^2          (c = SOFT / MEDIUM / HARD)

Setup:   pip install pandas numpy matplotlib
Usage:   python fit_degradation.py
Needs:   data/*_ALL_laps.csv  (from load_season.py)
Output:  models/tyre_model.json, tyre_degradation.png, and a printed summary
"""
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def load() -> pd.DataFrame:
    files = sorted(glob.glob("data/*_ALL_laps.csv"))
    if not files:
        raise SystemExit("No data/*_ALL_laps.csv found. Run load_season.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(["Race", "Driver", "LapNumber"], keep="last")
    if dup.any():
        print(f"Removed {int(dup.sum())} duplicate laps")
        df = df[~dup]
    df = df[df["clean"].astype(bool) & df["Compound"].isin(COMPOUNDS)]
    df = df.dropna(subset=["LapTimeSec", "TyreAge", "LapNumber"]).copy()
    df["grp"] = df["Race"].astype(str) + "|" + df["Driver"].astype(str)

    # Extra outlier guard on top of the 'clean' flag, then drop tiny groups.
    med = df.groupby("grp")["LapTimeSec"].transform("median")
    df = df[df["LapTimeSec"] <= 1.04 * med]
    n = df.groupby("grp")["LapTimeSec"].transform("size")
    return df[n >= 10].reset_index(drop=True)


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"fuel": df["LapNumber"].astype(float)}, index=df.index)
    for c in COMPOUNDS:
        m = (df["Compound"] == c).astype(float)
        X[f"{c}_age"] = m * df["TyreAge"]
        X[f"{c}_age2"] = m * df["TyreAge"] ** 2
    # A fresh soft is faster than a fresh hard. Without these offsets that pace gap leaks
    # into the wear slopes and makes soft tyres look like they wear less than they do.
    for c in COMPOUNDS[1:]:  # MEDIUM and HARD relative to SOFT
        X[f"{c}_off"] = (df["Compound"] == c).astype(float)
    return X


def demean(frame: pd.DataFrame, grp: pd.Series) -> pd.DataFrame:
    """Subtract each driver-race's own mean, which removes baseline pace."""
    return frame - frame.groupby(grp).transform("mean")


def fit(df: pd.DataFrame) -> pd.Series:
    y = demean(df[["LapTimeSec"]], df["grp"])["LapTimeSec"].to_numpy()
    X = demean(design(df), df["grp"])
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    return pd.Series(beta, index=X.columns)


def rmse_on(df: pd.DataFrame, beta: pd.Series):
    y = demean(df[["LapTimeSec"]], df["grp"])["LapTimeSec"].to_numpy()
    X = demean(design(df), df["grp"])
    pred = X.to_numpy() @ beta.to_numpy()
    return float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean((y - pred) ** 2)))


def loss(beta: pd.Series, compound: str, age):
    """Seconds lost versus a fresh tyre of this compound."""
    return beta[f"{compound}_age"] * age + beta[f"{compound}_age2"] * age ** 2


if __name__ == "__main__":
    df = load()
    races = df["Race"].unique()
    print(f"{len(df)} clean laps, {len(races)} races, {df['grp'].nunique()} driver-races")

    # Held-out check: fit on ~80% of races, test on the other ~20%.
    rng = np.random.default_rng(0)
    test_races = rng.choice(races, size=max(1, len(races) // 5), replace=False)
    train, test = df[~df["Race"].isin(test_races)], df[df["Race"].isin(test_races)]
    base, model = rmse_on(test, fit(train))
    print(f"\nHeld-out races ({len(test_races)}): RMSE without model {base:.3f}s, "
          f"with model {model:.3f}s  ({100 * (1 - model / base):.1f}% lower)")

    # Final fit on everything.
    beta = fit(df)
    p95 = {c: float(df.loc[df["Compound"] == c, "TyreAge"].quantile(0.95)) for c in COMPOUNDS}

    print(f"\nFuel/track-evolution slope: {beta['fuel']:+.4f} s per lap (negative = cars get faster)")
    print(f"Fresh-tyre pace vs soft: medium {beta['MEDIUM_off']:+.2f} s/lap, "
          f"hard {beta['HARD_off']:+.2f} s/lap")
    ages = [5, 10, 15, 20, 25, 30]
    print("\nTime lost vs a fresh tyre (seconds per lap):")
    print("age".ljust(8) + "".join(f"{a:>8}" for a in ages))
    for c in COMPOUNDS:
        cells = []
        for a in ages:
            cells.append(f"{loss(beta, c, a):8.2f}" if a <= p95[c] else "       -")
        n = int((df["Compound"] == c).sum())
        print(c.ljust(8) + "".join(cells) + f"   ({n} laps, 95% of laps under age {p95[c]:.0f})")
    print("('-' means beyond the tyre ages seen in the data, so we don't extrapolate)")

    Path("models").mkdir(exist_ok=True)
    out = {
        "fuel_s_per_lap": float(beta["fuel"]),
        "pace_offset_vs_soft": {"SOFT": 0.0, "MEDIUM": float(beta["MEDIUM_off"]),
                                "HARD": float(beta["HARD_off"])},
        "compounds": {c: {"lin": float(beta[f"{c}_age"]), "quad": float(beta[f"{c}_age2"]),
                          "max_age_p95": p95[c]} for c in COMPOUNDS},
        "n_laps": int(len(df)), "n_races": int(len(races)),
        "validation": {"rmse_without_model": base, "rmse_with_model": model},
    }
    Path("models/tyre_model.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "tab:gray"}
    for c in COMPOUNDS:
        x = np.linspace(0, p95[c], 50)
        ax.plot(x, loss(beta, c, x), label=c, color=colors[c], lw=2)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Time lost vs fresh tyre (s/lap)")
    ax.set_title("Fitted tyre degradation (fuel-corrected)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("tyre_degradation.png", dpi=150)
    print("\nSaved models/tyre_model.json and tyre_degradation.png")
