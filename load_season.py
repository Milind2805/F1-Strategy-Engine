"""
Download every race of a season from OpenF1 (reuses load_openf1.py).

Usage:   python load_season.py 2024
Output:  data/<year>_<Location>_laps.csv / _stints.csv  (one pair per race)
         data/<year>_ALL_laps.csv / _ALL_stints.csv      (combined, with Year and Race columns)
Resumable: races already saved are skipped, so you can stop and re-run safely.
Takes roughly 4-5 minutes per season because of the free-tier rate limit.
"""
import sys

import pandas as pd

from load_openf1 import DATA_DIR, build_tables, get


def main(year: int) -> None:
    sessions = get("sessions", year=year, session_name="Race")
    if sessions.empty:
        raise SystemExit(f"No races found for {year} (OpenF1 covers 2023 onwards).")
    sessions = sessions.sort_values("date_start")

    all_laps, all_stints = [], []
    for _, s in sessions.iterrows():
        tag = f"{year}_{str(s['location']).replace(' ', '_')}"
        lap_file = DATA_DIR / f"{tag}_laps.csv"
        stint_file = DATA_DIR / f"{tag}_stints.csv"

        if lap_file.exists() and stint_file.exists():
            laps_t, stints_t = pd.read_csv(lap_file), pd.read_csv(stint_file)
            print(f"cached   {tag}")
        else:
            key = int(s["session_key"])
            laps = get("laps", session_key=key)
            stints = get("stints", session_key=key)
            drivers = get("drivers", session_key=key)
            if laps.empty or stints.empty:
                print(f"skipped  {tag}: no laps/stints returned")
                continue
            try:
                laps_t, stints_t = build_tables(laps, stints, drivers)
            except Exception as e:  # one odd race should not stop the whole season
                print(f"skipped  {tag}: {e}")
                continue
            laps_t.to_csv(lap_file, index=False)
            stints_t.to_csv(stint_file, index=False)
            print(f"saved    {tag}: {int(laps_t['clean'].sum())} clean laps")

        all_laps.append(laps_t.assign(Year=year, Race=tag))
        all_stints.append(stints_t.assign(Year=year, Race=tag))

    if not all_laps:
        raise SystemExit("Nothing downloaded.")
    pd.concat(all_laps, ignore_index=True).to_csv(DATA_DIR / f"{year}_ALL_laps.csv", index=False)
    pd.concat(all_stints, ignore_index=True).to_csv(DATA_DIR / f"{year}_ALL_stints.csv", index=False)
    print(f"Done: {len(all_laps)} races combined for {year}.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
