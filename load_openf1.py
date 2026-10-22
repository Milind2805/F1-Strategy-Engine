"""
Phase 1 data loader using the free OpenF1 API (no key needed, seasons 2023+).
Fallback for when FastF1 cannot reach F1's live-timing servers.

Setup:   pip install requests pandas
Usage:   python load_openf1.py 2024 Italy        (place = country, city or circuit name)
Output:  data/<year>_<place>_laps.csv    (every lap, with a 'clean' flag)
         data/<year>_<place>_stints.csv  (one row per driver stint)
The columns match load_race.py, so later code works with either source.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get(endpoint: str, **params) -> pd.DataFrame:
    """GET one endpoint. The free tier allows ~30 requests/minute, so we pace ourselves."""
    for attempt in range(5):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429:  # rate limited: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:  # "no results"
            return pd.DataFrame()
        r.raise_for_status()
        time.sleep(2.2)
        return pd.DataFrame(r.json())
    raise RuntimeError(f"Rate limited too many times on {endpoint}")


def find_race(year: int, place: str) -> pd.Series:
    sessions = get("sessions", year=year, session_name="Race")
    if sessions.empty:
        raise SystemExit(f"No race sessions returned for {year}. OpenF1 covers 2023 onwards.")
    cols = [c for c in ("country_name", "location", "circuit_short_name") if c in sessions]
    text = sessions[cols].astype(str).agg(" ".join, axis=1).str.lower()
    match = sessions[text.str.contains(place.lower())]
    if match.empty:
        print("Available:", ", ".join(sorted(sessions["location"].astype(str))))
        raise SystemExit(f"No {year} race matching '{place}'.")
    return match.iloc[0]


def add_gap_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Seconds between each car and the next car ahead of it (same lap count) at the line.
    A small gap means dirty air or traffic, which slows a lap regardless of the tyres."""
    laps = laps.sort_values(["driver_number", "lap_number"]).copy()
    try:  # timestamps may or may not carry fractional seconds, so accept any ISO 8601 form
        start = pd.to_datetime(laps["date_start"], utc=True, errors="coerce", format="ISO8601")
    except ValueError:  # older pandas
        start = pd.to_datetime(laps["date_start"], utc=True, errors="coerce")
    next_start = start.groupby(laps["driver_number"]).shift(-1)  # the next lap starts when this one ends
    fallback = start + pd.to_timedelta(pd.to_numeric(laps["lap_duration"], errors="coerce"), unit="s")
    laps["t_end"] = next_start.fillna(fallback)
    laps = laps.sort_values(["lap_number", "t_end"])
    laps["gap_ahead"] = (laps["t_end"] - laps.groupby("lap_number")["t_end"].shift(1)).dt.total_seconds()
    return laps


def build_tables(laps: pd.DataFrame, stints: pd.DataFrame, drivers: pd.DataFrame):
    laps = laps.copy()
    if "date_start" in laps.columns:
        laps = add_gap_ahead(laps)
    else:
        laps["gap_ahead"] = float("nan")
    stints = stints.copy()
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].fillna(False).astype(bool)
    stints["lap_start"] = stints["lap_start"].fillna(1)
    stints["lap_end"] = stints["lap_end"].fillna(laps["lap_number"].max())

    keep = ["driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
    m = laps.merge(stints[keep], on="driver_number")
    m = m[(m["lap_number"] >= m["lap_start"]) & (m["lap_number"] <= m["lap_end"])].copy()
    # If two stints overlap on a lap, keep the later stint so each lap appears once.
    m = m.sort_values(["driver_number", "lap_number", "stint_number"])
    m = m.drop_duplicates(["driver_number", "lap_number"], keep="last")
    m["TyreAge"] = m["tyre_age_at_start"].fillna(0) + (m["lap_number"] - m["lap_start"])

    # In-lap = last lap of a stint that is followed by another stint. Out-lap = first lap after a stop.
    last_stint = m.groupby("driver_number")["stint_number"].transform("max")
    in_lap = (m["lap_number"] == m["lap_end"]) & (m["stint_number"] < last_stint)
    out_lap = m["is_pit_out_lap"] | ((m["lap_number"] == m["lap_start"]) & (m["stint_number"] > 1))

    # Clean = usable for tyre-wear modelling. The 107% rule is a crude safety-car filter;
    # we can replace it with race-control messages later.
    base = m["lap_duration"].notna() & ~in_lap & ~out_lap & (m["lap_number"] > 1)
    med = m["lap_duration"].where(base).groupby(m["driver_number"]).transform("median")
    m["clean"] = base & (m["lap_duration"] <= 1.07 * med)

    d = drivers[["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")
    m = m.merge(d, on="driver_number", how="left")
    out = pd.DataFrame({
        "Driver": m["name_acronym"],
        "Team": m["team_name"],
        "LapNumber": m["lap_number"].astype(int),
        "Stint": m["stint_number"].astype(int),
        "Compound": m["compound"],
        "TyreAge": m["TyreAge"],
        "LapTimeSec": m["lap_duration"],
        "clean": m["clean"],
        "GapAhead": m["gap_ahead"],
    }).sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    out["RaceLaps"] = int(out["LapNumber"].max())

    rows = []
    for (drv, st), g in out.groupby(["Driver", "Stint"]):
        c = g[g["clean"]]
        rows.append({
            "Driver": drv, "Stint": int(st), "Compound": g["Compound"].iloc[0],
            "StartLap": int(g["LapNumber"].min()), "EndLap": int(g["LapNumber"].max()),
            "NLaps": len(g), "StartTyreLife": g["TyreAge"].min(),
            "CleanLaps": len(c), "MedianCleanLapSec": c["LapTimeSec"].median(),
        })
    stint_table = pd.DataFrame(rows).sort_values(["Driver", "Stint"]).reset_index(drop=True)
    return out, stint_table


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    place = sys.argv[2] if len(sys.argv) > 2 else "Italy"

    race = find_race(year, place)
    key = int(race["session_key"])
    print(f"Found: {race.get('country_name')} / {race.get('location')} (session_key={key})")

    laps = get("laps", session_key=key)
    stints = get("stints", session_key=key)
    drivers = get("drivers", session_key=key)
    if laps.empty or stints.empty:
        raise SystemExit("OpenF1 returned no laps/stints for this session.")

    lap_table, stint_table = build_tables(laps, stints, drivers)

    tag = f"{year}_{place.replace(' ', '_')}"
    lap_table.to_csv(DATA_DIR / f"{tag}_laps.csv", index=False)
    stint_table.to_csv(DATA_DIR / f"{tag}_stints.csv", index=False)

    print(f"{lap_table['RaceLaps'].iloc[0]} laps, {lap_table['Driver'].nunique()} drivers, "
          f"{int(lap_table['clean'].sum())} clean laps")
    print(stint_table.head(15).to_string(index=False))
