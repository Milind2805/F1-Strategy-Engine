"""
Phase 1, step 4: circuit-specific tyre degradation.

The pooled model in fit_degradation.py treats every track alike, but tyre wear depends
heavily on the circuit. Here each circuit gets its own adjustment on top of the global
model, pulled back toward the global fit by a ridge penalty (lambda) so circuits with
little data don't produce noisy curves.

Validation: for every circuit with 2+ seasons, hold out its latest season, fit on the
earlier ones, and see whether the circuit-specific model beats the pooled model. Lambda
is picked from that same check.

Usage:   python fit_degradation_by_circuit.py      (needs data/*_ALL_laps.csv)
Output:  models/tyre_model_by_circuit.json and a printed summary
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_degradation import COMPOUNDS, demean, design, fit, load

LAMBDAS = [100, 500, 2000, 10000, 50000, 200000]
MIN_LAPS = 300  # per circuit and compound, below this the curve is flagged unreliable


def cdesign(df: pd.DataFrame) -> pd.DataFrame:
    """Circuit-level adjustment features (age scaled by 10 so the penalty treats them evenly)."""
    X = pd.DataFrame({"lap": df["LapNumber"].astype(float) / 10}, index=df.index)
    u = df["TyreAge"] / 10
    for c in COMPOUNDS:
        m = (df["Compound"] == c).astype(float)
        X[f"{c}_u"] = m * u
        X[f"{c}_u2"] = m * u ** 2
    for c in COMPOUNDS[1:]:  # circuit-specific fresh-tyre pace gap vs soft
        X[f"{c}_off"] = (df["Compound"] == c).astype(float)
    return X


def global_residual(df: pd.DataFrame, beta: pd.Series) -> pd.Series:
    y = demean(df[["LapTimeSec"]], df["grp"])["LapTimeSec"]
    return y - demean(design(df), df["grp"]) @ beta


def fit_circuits(train: pd.DataFrame, beta: pd.Series, lam: float) -> dict:
    r = global_residual(train, beta)
    deltas = {}
    for k, g in train.groupby("Circuit"):
        X = demean(cdesign(g), g["grp"])
        A = X.to_numpy().T @ X.to_numpy() + lam * np.eye(X.shape[1])
        d = np.linalg.pinv(A) @ (X.to_numpy().T @ r.loc[g.index].to_numpy())
        deltas[k] = pd.Series(d, index=X.columns)
    return deltas


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def evaluate(test: pd.DataFrame, beta: pd.Series, deltas: dict):
    r = global_residual(test, beta)
    err = r.copy()
    for k, g in test.groupby("Circuit"):
        if k in deltas:
            X = demean(cdesign(g), g["grp"]).to_numpy()
            err.loc[g.index] = r.loc[g.index].to_numpy() - X @ deltas[k].to_numpy()
    y = demean(test[["LapTimeSec"]], test["grp"])["LapTimeSec"]
    return rmse(y), rmse(r), rmse(err)


def circuit_params(beta: pd.Series, d: pd.Series, df_c: pd.DataFrame, p95_global: dict) -> dict:
    out = {"fuel_s_per_lap": float(beta["fuel"] + d["lap"] / 10),
           "pace_offset_vs_soft": {"SOFT": 0.0,
                                   "MEDIUM": float(beta["MEDIUM_off"] + d["MEDIUM_off"]),
                                   "HARD": float(beta["HARD_off"] + d["HARD_off"])},
           "compounds": {}}
    for c in COMPOUNDS:
        sub = df_c[df_c["Compound"] == c]
        out["compounds"][c] = {
            "lin": float(beta[f"{c}_age"] + d[f"{c}_u"] / 10),
            "quad": float(beta[f"{c}_age2"] + d[f"{c}_u2"] / 100),
            "max_age_p95": float(sub["TyreAge"].quantile(0.95)) if len(sub) >= MIN_LAPS else p95_global[c],
            "n_laps": int(len(sub)),
            "reliable": bool(len(sub) >= MIN_LAPS),
        }
    return out


if __name__ == "__main__":
    df = load()
    df["Year"] = df["Race"].str.split("_").str[0].astype(int)
    df["Circuit"] = df["Race"].str.split("_", n=1).str[1]
    print(f"{len(df)} clean laps, {df['Circuit'].nunique()} circuits, years {sorted(int(y) for y in df['Year'].unique())}")

    last = df.groupby("Circuit")["Year"].transform("max")
    first = df.groupby("Circuit")["Year"].transform("min")
    test = df[(df["Year"] == last) & (last > first)]
    train = df.drop(test.index)
    print(f"Validation: hold out the latest season of {test['Circuit'].nunique()} circuits "
          f"({len(test)} laps), train on the rest\n")

    beta_tr = fit(train)
    print(f"{'lambda':>8} {'no model':>10} {'pooled':>8} {'by circuit':>11}")
    best_lam, best_err = None, 1e9
    for lam in LAMBDAS:
        base, pooled, circ = evaluate(test, beta_tr, fit_circuits(train, beta_tr, lam))
        print(f"{lam:>8} {base:>9.3f}s {pooled:>7.3f}s {circ:>10.3f}s")
        if circ < best_err:
            best_lam, best_err = lam, circ
    print(f"\nBest lambda: {best_lam}.  Circuit-specific model is "
          f"{100 * (1 - best_err / pooled):.1f}% lower error than the pooled model on unseen races.")

    # Final fit on everything.
    beta = fit(df)
    deltas = fit_circuits(df, beta, best_lam)
    p95_global = {c: float(df.loc[df["Compound"] == c, "TyreAge"].quantile(0.95)) for c in COMPOUNDS}
    params = {k: circuit_params(beta, d, df[df["Circuit"] == k], p95_global) for k, d in deltas.items()}

    def loss(p, c, age):
        return p["compounds"][c]["lin"] * age + p["compounds"][c]["quad"] * age ** 2

    print("\nTime lost vs fresh tyre at tyre age 15 (s/lap), sorted by medium-tyre wear:")
    print(f"{'circuit':<22}{'SOFT':>7}{'MED':>7}{'HARD':>7}   {'M-pace':>7}{'H-pace':>7}")
    rows = []
    for k, p in params.items():
        cells = []
        for c in COMPOUNDS:
            ok = p["compounds"][c]["reliable"] and p["compounds"][c]["max_age_p95"] >= 15
            cells.append(loss(p, c, 15) if ok else np.nan)
        rows.append((k, cells, (p["pace_offset_vs_soft"]["MEDIUM"], p["pace_offset_vs_soft"]["HARD"])))
    rows.sort(key=lambda r: -np.nan_to_num(r[1][1], nan=-1))
    for k, cells, (mo, ho) in rows:
        txt = "".join(f"{v:7.2f}" if not np.isnan(v) else "      -" for v in cells)
        print(f"{k:<22}{txt}   {mo:+7.2f}{ho:+7.2f}")
    print("('-' = too few laps at that tyre age. M-pace/H-pace = fresh medium/hard pace vs fresh soft, s/lap)")

    Path("models").mkdir(exist_ok=True)
    Path("models/tyre_model_by_circuit.json").write_text(json.dumps({
        "global": {"fuel_s_per_lap": float(beta["fuel"]),
                   "compounds": {c: {"lin": float(beta[f"{c}_age"]), "quad": float(beta[f"{c}_age2"]),
                                     "max_age_p95": p95_global[c]} for c in COMPOUNDS}},
        "circuits": params,
        "lambda": best_lam,
        "validation": {"best_rmse_by_circuit": best_err, "rmse_pooled": pooled, "rmse_no_model": base},
    }, indent=2))
    print("\nSaved models/tyre_model_by_circuit.json")
