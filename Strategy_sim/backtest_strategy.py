"""
Phase 1, step 7: backtest the strategy simulator against what drivers really did.

For every dry race of a season it:
  1. finds the strategy most finishers actually ran (compound order and median stint lengths),
  2. runs the simulator on that circuit and picks the best of ~thousands of candidate strategies,
  3. compares the two.

IMPORTANT: the real strategy is simulated on its own and is NOT among the candidates. (An earlier
version added it to the candidate list, so whenever it beat every grid strategy the simulator
"agreed" with it by construction. That flattered the agreement numbers.)

Two modes:
  default   a random safety car in every simulated race (no reactions): "plan in advance"
  --replay  the safety car comes on the laps it really did in that race, and the simulator's plans
            may pit under it (see SimConfig.react_*). The real strategy is simulated as it was
            actually run, so its stops already contain the teams' own reactions.

Metrics:
  * stop-count agreement with the majority of finishers
  * compound agreement (same set of compounds, ignoring order)
  * regret = real strategy's average race time minus the simulator's best, in simulation.
    Negative means the real strategy beat every candidate.
  * rank = share of candidate strategies that beat the real one (small is good for the real one)

Wet or mixed races are skipped, and so are chaotic ones (a stint under 3 laps) and, in replay mode,
red-flag races.

Usage:   python backtest_strategy.py 2025 --holdout            (needs: python fit_degradation_fe.py 2025)
         python backtest_strategy.py 2025 --holdout --replay
Output:  backtest_<year>[_holdout][_replay].csv and .md
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_strategy import (SimConfig, build_draws, circuit_caps, circuit_files,
                               detect_safety_car_laps, estimate_pit_loss, label, load_model,
                               simulate_all, strategies_with_fallback)

LETTER = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}


def modal_strategy(stints: pd.DataFrame, L: int):
    """Most common compound sequence among drivers who finished (near) full distance."""
    seqs, stops = {}, []
    for _, g in stints.sort_values(["Driver", "Stint"]).groupby("Driver"):
        if g["EndLap"].max() < L - 2:
            continue
        letters = "".join(c[0] for c in g["Compound"])
        seqs.setdefault(letters, []).append(g["NLaps"].tolist())
        stops.append(len(g) - 1)
    if not seqs:
        return None
    letters = max(seqs, key=lambda k: len(seqs[k]))
    lens = np.median(np.array(seqs[letters], dtype=float), axis=0).round().astype(int)
    lens[-1] = L - lens[:-1].sum()
    if len(letters) < 2 or (lens < 1).any():
        return None
    strat = tuple((LETTER[ch], int(n)) for ch, n in zip(letters, lens))
    majority_stops = max(set(stops), key=stops.count)
    return strat, letters, majority_stops, len(seqs[letters]), len(stops)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    year = int(args[0]) if args else 2025
    holdout, replay = "--holdout" in sys.argv, "--replay" in sys.argv

    if holdout:
        model_path = f"models/tyre_model_fe_holdout{year}.json"
        print(f"Using the tyre model fitted WITHOUT {year} (clean out-of-sample test).")
    else:
        model_path = "models/tyre_model_fe.json"
        print(f"Note: this tyre model was fitted on all seasons including {year}, so results are optimistic. "
              f"For a clean test run: python fit_degradation_fe.py {year}, then add --holdout.")
    print("Mode: " + ("replay of the real safety-car laps, simulator plans may pit under it"
                      if replay else "random safety car, plans fixed in advance"))
    B, global_caps, trend = load_model(model_path)
    cfg = SimConfig(react_to_sc=replay)

    rows = []
    for lap_file in sorted(glob.glob(f"data/{year}_*_laps.csv")):
        if "_ALL_" in lap_file:
            continue
        circuit = Path(lap_file).name[len(f"{year}_"):-len("_laps.csv")]
        stint_file = Path(f"data/{year}_{circuit}_stints.csv")
        if not stint_file.exists():
            continue
        stints = pd.read_csv(stint_file)
        laps = pd.read_csv(lap_file).drop_duplicates(["Driver", "LapNumber"], keep="last")

        if (~stints["Compound"].isin(list(LETTER.values()))).mean() > 0.05:
            print(f"skipped  {circuit}: wet or mixed conditions")
            continue
        L = int(laps["RaceLaps"].iloc[0])
        modal = modal_strategy(stints, L)
        if modal is None:
            print(f"skipped  {circuit}: no usable finisher strategies")
            continue
        actual, letters, majority_stops, n_same, n_fin = modal
        if min(n for _, n in actual) < 3:
            print(f"skipped  {circuit}: chaotic race (a stint under 3 laps)")
            continue

        ref = float(laps.loc[laps["clean"].astype(bool), "LapTimeSec"].median())
        sc_laps, red_flag = detect_safety_car_laps(laps, ref)
        if replay and red_flag:
            print(f"skipped  {circuit}: red flag")
            continue

        caps, _ = circuit_caps(circuit, year, global_caps)  # learned from other seasons only
        files = [f for f in circuit_files(circuit, "laps") if not Path(f).name.startswith(f"{year}_")]
        pit_loss, _ = estimate_pit_loss(files)

        rng = np.random.default_rng(42)
        d = build_draws(B, L, rng, cfg, sc_laps=sc_laps if replay else None)
        candidates, caps_used, relaxed = strategies_with_fallback(L, caps, global_caps)
        if not candidates:
            print(f"skipped  {circuit}: no feasible strategy within the stint limits")
            continue
        res = simulate_all(candidates, d, L, ref, trend, pit_loss, cfg, caps=caps_used, react=replay)
        res_real = simulate_all([actual], d, L, ref, trend, pit_loss, cfg, react=False)

        mean = res.mean(axis=1)
        best = int(np.argmin(mean))
        mean_real = float(res_real.mean())
        regret = mean_real - float(mean[best])
        rank_pct = float(100 * (mean < mean_real).mean())
        best_letters = "".join(c[0] for c, _ in candidates[best])
        rows.append({
            "circuit": circuit, "laps": L, "sim_best": label(candidates[best]), "actual_modal": label(actual),
            "finishers_on_it": f"{n_same}/{n_fin}", "safety_car_laps": len(sc_laps), "limits_widened": relaxed,
            "sim_stops": len(candidates[best]) - 1, "actual_stops": majority_stops,
            "stops_match": (len(candidates[best]) - 1) == majority_stops,
            "compounds_match": sorted(best_letters) == sorted(letters),
            "regret_s": round(regret, 1), "real_rank_pct": round(rank_pct, 1),
        })
        print(f"done     {circuit}: sim {label(candidates[best])} | real {label(actual)} | "
              f"regret {regret:+.1f}s | SC laps {len(sc_laps)}" + (" | limits widened" if relaxed else ""))

    if not rows:
        raise SystemExit("No races evaluated.")
    out = pd.DataFrame(rows)
    tag = f"{year}{'_holdout' if holdout else ''}{'_replay' if replay else ''}"
    out.to_csv(f"backtest_{tag}.csv", index=False)

    summary = [
        f"{len(out)} dry races evaluated for {year}",
        f"Stop count matches the majority of finishers: {100 * out['stops_match'].mean():.0f}%",
        f"Same set of compounds as the real strategy:   {100 * out['compounds_match'].mean():.0f}%",
        f"Regret of the real strategy in simulation:    median {out['regret_s'].median():+.1f}s, "
        f"{100 * (out['regret_s'] <= 5).mean():.0f}% of races within 5s, "
        f"{100 * (out['regret_s'] <= 10).mean():.0f}% within 10s",
        f"Real strategy beat every candidate in {int((out['real_rank_pct'] == 0).sum())} races; "
        f"median share of candidates that beat it: {out['real_rank_pct'].median():.0f}%",
    ]
    print("\n" + "\n".join(summary))
    md = (f"# Backtest {tag}\n\n"
          f"Mode: {'replay of real safety-car laps' if replay else 'random safety car'}; "
          f"tyre model: {'fitted without ' + str(year) if holdout else 'fitted on all seasons (optimistic)'}.\n\n"
          + "\n".join(f"- {line}" for line in summary)
          + "\n\n```\n" + out.to_string(index=False) + "\n```\n")
    Path(f"backtest_{tag}.md").write_text(md, encoding="utf-8")
    print(f"\nSaved backtest_{tag}.csv")
