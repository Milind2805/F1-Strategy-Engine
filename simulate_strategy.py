"""
Phase 1, step 6: Monte Carlo pit-stop strategy simulator.

For one circuit it enumerates 1-stop and 2-stop strategies, simulates each one many times
under uncertainty, and ranks them by expected race time.

Uncertainty included in every simulated race:
  * tyre-model uncertainty (each run uses one of the bootstrap fits from fit_degradation_fe.py)
  * lap-to-lap noise
  * pit-stop time variation
  * a random safety car (cheaper pit stops while it is out)
All strategies are simulated on the SAME random draws, so differences between them are fair.

Usage:   python simulate_strategy.py Monza 2025
Needs:   models/tyre_model_fe.json (run fit_degradation_fe.py first) and data/<year>_<circuit>_*.csv

Limits (v1): one car alone on track (no traffic or opponents), one pooled tyre model for all
circuits, simple safety-car model. Treat results as a starting point, not a verdict.
"""
import glob
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
FEATURES = ["SOFT_age", "SOFT_age2", "MEDIUM_age", "MEDIUM_age2", "HARD_age", "HARD_age2",
            "MEDIUM_off", "HARD_off"]

N_SIMS = 400
LAP_NOISE_S = 0.40        # lap-to-lap noise (s)
PIT_NOISE_S = 0.8         # pit-stop time variation (s)
P_SAFETY_CAR = 0.5        # assumption: chance a race has one safety car
SC_LAPS = 4               # laps the safety car stays out
SC_LAP_FACTOR = 1.40      # safety-car laps are 40% slower
SC_PIT_DISCOUNT = 0.45    # a stop under the safety car costs 45% of the normal loss
DEFAULT_PIT_LOSS = 22.0   # used if the data gives too few clean pit stops
MIN_STINT = 8
MAX_STINT_FACTOR = 1.15   # no stint longer than 1.15x the longest typical age seen in the data


@dataclass(frozen=True)
class SimConfig:
    """Simulation settings. Defaults come from the module constants above."""
    n_sims: int = N_SIMS
    lap_noise_s: float = LAP_NOISE_S
    pit_noise_s: float = PIT_NOISE_S
    p_safety_car: float = P_SAFETY_CAR
    sc_laps: int = SC_LAPS
    sc_lap_factor: float = SC_LAP_FACTOR
    sc_pit_discount: float = SC_PIT_DISCOUNT


DEFAULT_CONFIG = SimConfig()


def load_model(path="models/tyre_model_fe.json"):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run fit_degradation_fe.py "
                         f"{'<year> ' if 'holdout' in path.name else ''}first.")
    m = json.loads(path.read_text())
    if m.get("boot_betas") and m.get("features") == FEATURES:
        B = np.array(m["boot_betas"], dtype=float)
    else:
        print("Note: no bootstrap draws in the model file, so model uncertainty is skipped. "
              "Re-run fit_degradation_fe.py to include it.")
        point = [m["compounds"][c][k] for c in COMPOUNDS for k in ("lin", "quad")]
        point += [m["pace_offset_vs_soft"]["MEDIUM"], m["pace_offset_vs_soft"]["HARD"]]
        B = np.array([point], dtype=float)
    caps = {c: int(np.ceil(MAX_STINT_FACTOR * m["compounds"][c]["max_age_p95"])) for c in COMPOUNDS}
    return B, caps, float(m["fuel_s_per_lap"])


def estimate_pit_loss(files):
    """Median time lost per stop = in-lap + out-lap - 2 x the driver's normal pace nearby."""
    losses = []
    for f in files:
        laps = pd.read_csv(f)
        for _, g in laps.groupby("Driver"):
            g = g.sort_values("LapNumber").drop_duplicates("LapNumber", keep="last").set_index("LapNumber")
            t, clean = g["LapTimeSec"], g["clean"].astype(bool)
            for s in sorted(g["Stint"].unique())[:-1]:
                in_lap = int(g.index[g["Stint"] == s].max())
                out_lap = in_lap + 1
                if out_lap not in g.index or np.isnan(t.get(in_lap, np.nan)) or np.isnan(t.get(out_lap, np.nan)):
                    continue
                before = [l for l in range(in_lap - 4, in_lap) if l in g.index and clean[l]]
                after = [l for l in range(out_lap + 1, out_lap + 5) if l in g.index and clean[l]]
                if len(before) < 2 or len(after) < 2:
                    continue
                ref = np.mean([t[l] for l in before + after])
                loss = t[in_lap] + t[out_lap] - 2 * ref
                if 12 < loss < 40:
                    losses.append(loss)
    if len(losses) >= 8:
        return float(np.median(losses)), len(losses)
    return DEFAULT_PIT_LOSS, len(losses)


def circuit_caps(circuit, exclude_year, global_caps):
    """Longest stint we allow per compound at this circuit, learned from how long teams actually
    ran each compound there in other seasons (so the season being tested is never used)."""
    lens = {c: [] for c in COMPOUNDS}
    for f in glob.glob(f"data/*_{circuit}_stints.csv"):
        if Path(f).name.startswith(f"{exclude_year}_"):
            continue
        st = pd.read_csv(f)
        for c in COMPOUNDS:
            lens[c] += st.loc[st["Compound"] == c, "NLaps"].tolist()
    caps = {}
    for c in COMPOUNDS:
        v = np.array(lens[c], dtype=float)
        if len(v) >= 8:
            cap = int(np.ceil(np.percentile(v, 95))) + 2
        elif len(v) > 0:
            cap = int(min(global_caps[c], v.max() + 2))
        else:
            cap = int(min(global_caps[c], 15))  # never seen here: be conservative
        caps[c] = max(cap, MIN_STINT)
    return caps, {c: len(lens[c]) for c in COMPOUNDS}


def enumerate_strategies(L, caps):
    out = []
    for comps in itertools.product(COMPOUNDS, repeat=2):
        if len(set(comps)) < 2:
            continue
        for p in range(MIN_STINT, L - MIN_STINT + 1, 2):
            st = ((comps[0], p), (comps[1], L - p))
            if all(n <= caps[c] for c, n in st):
                out.append(st)
    for comps in itertools.product(COMPOUNDS, repeat=3):
        if len(set(comps)) < 2:
            continue
        for p1 in range(MIN_STINT, L - 2 * MIN_STINT + 1, 3):
            for p2 in range(p1 + MIN_STINT, L - MIN_STINT + 1, 3):
                st = ((comps[0], p1), (comps[1], p2 - p1), (comps[2], L - p2))
                if all(n <= caps[c] for c, n in st):
                    out.append(st)
    return out


def build_draws(B, L, rng, cfg=DEFAULT_CONFIG):
    n = cfg.n_sims
    Bs = B[rng.integers(0, len(B), n)]
    d = {
        "lin": {"SOFT": Bs[:, 0], "MEDIUM": Bs[:, 2], "HARD": Bs[:, 4]},
        "quad": {"SOFT": Bs[:, 1], "MEDIUM": Bs[:, 3], "HARD": Bs[:, 5]},
        "off": {"SOFT": np.zeros(n), "MEDIUM": Bs[:, 6], "HARD": Bs[:, 7]},
        "noise": rng.normal(0, cfg.lap_noise_s, (n, L)),
        "pit_noise": rng.normal(0, cfg.pit_noise_s, (n, 3)),
    }
    has_sc = rng.random(n) < cfg.p_safety_car
    start = rng.integers(2, max(3, L - cfg.sc_laps - 1), n)
    laps = np.arange(1, L + 1)[None, :]
    d["sc"] = has_sc[:, None] & (laps >= start[:, None]) & (laps < start[:, None] + cfg.sc_laps)
    # Extra pit-stop noise for strategies with more than 3 stops (drawn last, so the first
    # draws, and therefore earlier results, are unchanged).
    d["pit_noise"] = np.hstack([d["pit_noise"], rng.normal(0, cfg.pit_noise_s, (n, 5))])
    return d


def simulate_all(strats, d, L, ref, trend, pit_loss, cfg=DEFAULT_CONFIG):
    n = cfg.n_sims
    idx = np.arange(L)
    res = np.empty((len(strats), n), dtype=np.float32)
    for i, strat in enumerate(strats):
        comp = np.repeat([c for c, _ in strat], [m for _, m in strat])
        age = np.concatenate([np.arange(m) for _, m in strat]).astype(float)
        wear = np.zeros((n, L))
        offs = np.zeros((n, L))
        for c in COMPOUNDS:
            mask = comp == c
            if mask.any():
                wear[:, mask] = d["lin"][c][:, None] * age[mask] + d["quad"][c][:, None] * age[mask] ** 2
                offs[:, mask] = d["off"][c][:, None]
        t = ref + trend * idx + offs + wear + d["noise"]
        t = np.where(d["sc"], cfg.sc_lap_factor * ref, t)
        pit = np.zeros(n)
        for k, s_lap in enumerate(np.cumsum([m for _, m in strat])[:-1]):
            f = np.where(d["sc"][:, s_lap - 1], cfg.sc_pit_discount, 1.0)
            pit += (pit_loss + d["pit_noise"][:, k]) * f
        res[i] = t.sum(axis=1) + pit
    return res


def label(strat):
    return " > ".join(f"{c[0]}{n}" for c, n in strat)


def actual_strategies(stint_file):
    s = pd.read_csv(stint_file)
    seqs, stops = Counter(), Counter()
    for _, g in s.groupby("Driver"):
        g = g.sort_values("Stint")
        seqs["".join(c[0] for c in g["Compound"])] += 1
        stops[len(g) - 1] += 1
    return seqs, stops


if __name__ == "__main__":
    circuit = sys.argv[1] if len(sys.argv) > 1 else "Monza"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    lap_file = Path(f"data/{year}_{circuit}_laps.csv")
    stint_file = Path(f"data/{year}_{circuit}_stints.csv")
    if not lap_file.exists():
        names = sorted({Path(p).name.split("_", 1)[1].rsplit("_laps", 1)[0]
                        for p in glob.glob(f"data/{year}_*_laps.csv") if "_ALL_" not in p})
        raise SystemExit(f"No data for {year} {circuit}. Available: {', '.join(names)}")

    holdout_model = f"models/tyre_model_fe_holdout{year}.json"
    B, global_caps, trend = load_model(holdout_model if "--holdout" in sys.argv else "models/tyre_model_fe.json")
    caps, n_obs = circuit_caps(circuit, year, global_caps)
    laps = pd.read_csv(lap_file).drop_duplicates(["Driver", "LapNumber"], keep="last")
    L = int(laps["RaceLaps"].iloc[0])
    ref = float(laps.loc[laps["clean"].astype(bool), "LapTimeSec"].median())
    files = [f for f in glob.glob(f"data/*_{circuit}_laps.csv")
             if "_ALL_" not in f and not Path(f).name.startswith(f"{year}_")]
    pit_loss, n_stops = estimate_pit_loss(files)
    print(f"{circuit} {year}: {L} laps, typical lap {ref:.1f}s, pit-stop loss {pit_loss:.1f}s "
          f"(from {n_stops} clean stops{'; default used' if n_stops < 8 else ''})")
    print("Max stint lengths allowed (from other seasons here): "
          + ", ".join(f"{c} {n} ({n_obs[c]} stints seen)" for c, n in caps.items()))
    print(f"(Pit loss and stint limits use seasons other than {year}, as if the race had not happened yet.)")

    rng = np.random.default_rng(42)
    d = build_draws(B, L, rng)
    strats = enumerate_strategies(L, caps)
    print(f"Simulating {len(strats)} strategies x {N_SIMS} races each...")
    res = simulate_all(strats, d, L, ref, trend, pit_loss)

    mean = res.mean(axis=1)
    order = np.argsort(mean)
    best = order[0]
    print("\nTop strategies (letters = compound, number = laps on that set):")
    print(f"{'#':>2}  {'strategy':<20}{'vs best':>9}{'  10-90% range':>16}{'  beats #1':>11}")
    for rank, i in enumerate(order[:10], 1):
        delta = res[i] - res[best]
        lo, hi = np.percentile(delta, [10, 90])
        print(f"{rank:>2}  {label(strats[i]):<20}{mean[i] - mean[best]:>+8.1f}s"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>16}{100 * (res[i] < res[best]).mean():>10.0f}%")

    for n_st in (1, 2):
        idxs = [i for i, s in enumerate(strats) if len(s) == n_st + 1]
        if idxs:
            j = min(idxs, key=lambda i: mean[i])
            print(f"\nBest {n_st}-stop: {label(strats[j])}  ({mean[j] - mean[best]:+.1f}s vs overall best)")

    if stint_file.exists():
        seqs, stops = actual_strategies(stint_file)
        print(f"\nWhat the drivers actually did in {year} {circuit}:")
        print("  stops: " + ", ".join(f"{k}-stop x{v}" for k, v in sorted(stops.items())))
        print("  compound sequences: " + ", ".join(f"{k} x{v}" for k, v in seqs.most_common(5)))
    print("\nLimits: single car, no traffic/opponents, pooled tyre model, simplified safety car.")
