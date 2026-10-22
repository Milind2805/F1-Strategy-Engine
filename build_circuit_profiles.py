"""
Summarise data/ into models/circuit_profiles.json.

The demo app (and anyone cloning the repo) can then run without the raw lap files. For each
circuit it stores race length, typical lap time, pit-stop time loss, the longest stint teams
have run on each compound there, and what drivers did in the latest dry race.

Usage:   python build_circuit_profiles.py
Needs:   data/<year>_<circuit>_laps.csv and _stints.csv (from load_season.py),
         models/tyre_model_fe.json (from fit_degradation_fe.py)
Output:  models/circuit_profiles.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_strategy import (COMPOUNDS, actual_strategies, circuit_caps, estimate_pit_loss,
                               load_model)


def main() -> None:
    files = [f for f in sorted(glob.glob("data/*_laps.csv")) if "_ALL_" not in f]
    if not files:
        raise SystemExit("No data/*_laps.csv files found. Run load_season.py first.")

    by_circuit = {}
    for f in files:
        name = Path(f).name
        year = int(name.split("_", 1)[0])
        circuit = name[len(f"{year}_"):-len("_laps.csv")]
        by_circuit.setdefault(circuit, []).append((year, f))

    _, global_caps, _ = load_model()
    profiles = {}
    for circuit, items in sorted(by_circuit.items()):
        items.sort()
        race_laps, typical, actual = [], [], None
        for year, f in items:
            laps = pd.read_csv(f).drop_duplicates(["Driver", "LapNumber"], keep="last")
            race_laps.append(int(laps["RaceLaps"].iloc[0]))
            clean = laps.loc[laps["clean"].astype(bool), "LapTimeSec"]
            if len(clean):
                typical.append(float(clean.median()))
            stint_file = Path(f"data/{year}_{circuit}_stints.csv")
            if stint_file.exists():
                stints = pd.read_csv(stint_file)
                if stints["Compound"].isin(COMPOUNDS).mean() > 0.95:  # dry race only
                    seqs, stops = actual_strategies(stint_file)
                    actual = {
                        "year": year,
                        "stops": {str(k): int(v) for k, v in sorted(stops.items())},
                        "sequences": {k: int(v) for k, v in seqs.most_common(5)},
                    }
        if not typical:
            continue
        pit_loss, n_stops = estimate_pit_loss([f for _, f in items])
        caps, n_seen = circuit_caps(circuit, None, global_caps)
        profiles[circuit] = {
            "years": [y for y, _ in items],
            "race_laps": max(race_laps),
            "typical_lap_s": float(np.median(typical)),
            "pit_loss_s": float(pit_loss),
            "pit_loss_stops_used": int(n_stops),
            "stint_caps": {c: int(v) for c, v in caps.items()},
            "stints_seen": {c: int(v) for c, v in n_seen.items()},
            "actual": actual,
        }

    Path("models").mkdir(exist_ok=True)
    Path("models/circuit_profiles.json").write_text(
        json.dumps({"circuits": profiles}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved models/circuit_profiles.json ({len(profiles)} circuits)")


if __name__ == "__main__":
    main()
