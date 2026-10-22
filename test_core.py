"""Fast tests on synthetic data (no network, no real data needed)."""
import numpy as np
import pandas as pd
import pytest

import load_openf1 as lo
import simulate_strategy as sim

CAPS = {"SOFT": 25, "MEDIUM": 34, "HARD": 40}
B = np.array([[0.10, 0.004, 0.05, 0.002, 0.025, 0.001, 0.3, 0.6],
              [0.12, 0.003, 0.06, 0.001, 0.030, 0.0008, 0.25, 0.5]])


def test_strategies_follow_the_rules():
    L = 53
    strats = sim.enumerate_strategies(L, CAPS)
    assert strats
    for s in strats:
        assert sum(n for _, n in s) == L                      # covers the whole race
        assert len({c for c, _ in s}) >= 2                     # two different compounds
        assert all(sim.MIN_STINT <= n <= CAPS[c] for c, n in s)
        assert len(s) in (2, 3)                                # 1-stop or 2-stop


def test_simulation_is_reproducible_and_fair():
    L = 53
    strats = sim.enumerate_strategies(L, CAPS)[:50]
    runs = []
    for _ in range(2):
        d = sim.build_draws(B, L, np.random.default_rng(1))
        runs.append(sim.simulate_all(strats, d, L, 83.0, -0.05, 23.0))
    assert np.array_equal(runs[0], runs[1])


def test_more_wear_means_slower_race():
    L = 53
    d = sim.build_draws(B, L, np.random.default_rng(1))
    s = [(("MEDIUM", 26), ("HARD", 27))]
    base = sim.simulate_all(s, d, L, 83.0, -0.05, 23.0).mean()
    worse = dict(d)
    worse["lin"] = {c: v * 3 for c, v in d["lin"].items()}
    assert sim.simulate_all(s, worse, L, 83.0, -0.05, 23.0).mean() > base


def test_config_changes_results_without_touching_globals():
    L = 40
    cfg = sim.SimConfig(n_sims=50, p_safety_car=1.0)
    d = sim.build_draws(B, L, np.random.default_rng(3), cfg)
    assert d["sc"].shape == (50, L) and d["sc"].any(axis=1).all()
    assert sim.N_SIMS == 400  # module default is untouched


def test_pit_loss_is_recovered(tmp_path):
    rows = []
    for drv in ("A", "B", "C"):
        for lap in range(1, 41):
            stint = 1 if lap <= 20 else 2
            t = 90.0
            if lap == 20:
                t += 10      # in-lap
            if lap == 21:
                t += 12      # out-lap  -> total loss 22 s
            rows.append(dict(Driver=drv, LapNumber=lap, Stint=stint, LapTimeSec=t,
                             clean=lap not in (20, 21), Compound="MEDIUM" if stint == 1 else "HARD"))
    f = tmp_path / "2025_X_laps.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    # one file has only 3 stops, below the 8-stop minimum, so duplicate it under other names
    files = []
    for i in range(3):
        g = tmp_path / f"202{i}_X_laps.csv"
        pd.DataFrame(rows).to_csv(g, index=False)
        files.append(str(g))
    loss, n = sim.estimate_pit_loss(files)
    assert n >= 8 and loss == pytest.approx(22.0, abs=0.5)


def test_pit_loss_falls_back_to_default_when_data_is_thin(tmp_path):
    loss, n = sim.estimate_pit_loss([])
    assert loss == sim.DEFAULT_PIT_LOSS and n == 0


def test_circuit_caps_ignore_the_tested_season(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    for year, length in ((2024, 20), (2025, 45)):
        pd.DataFrame({"Compound": ["SOFT"] * 10, "NLaps": [length] * 10}).to_csv(
            tmp_path / "data" / f"{year}_Z_stints.csv", index=False)
    caps, seen = sim.circuit_caps("Z", 2025, CAPS)
    assert caps["SOFT"] == 22 and seen["SOFT"] == 10     # 2025 data (45 laps) was not used


def test_gap_to_car_ahead():
    t0 = pd.Timestamp("2024-05-01 13:00:00", tz="UTC")
    rows = []
    for lap in range(1, 5):
        base = t0 + pd.Timedelta(seconds=90 * (lap - 1))
        for drv, lag in ((1, 0.0), (2, 0.8), (3, 6.8)):
            rows.append(dict(driver_number=drv, lap_number=lap,
                             date_start=(base + pd.Timedelta(seconds=lag)).isoformat(), lap_duration=90.0))
    g = lo.add_gap_ahead(pd.DataFrame(rows))
    lap2 = g[g.lap_number == 2].set_index("driver_number")["gap_ahead"]
    assert np.isnan(lap2[1]) and lap2[2] == pytest.approx(0.8) and lap2[3] == pytest.approx(6.0)


def test_overlapping_stints_do_not_duplicate_laps():
    laps = pd.DataFrame({"driver_number": [1] * 6, "lap_number": range(1, 7),
                         "lap_duration": [90.0] * 6, "is_pit_out_lap": [False] * 6})
    stints = pd.DataFrame({"driver_number": [1, 1], "stint_number": [1, 2], "lap_start": [1, 3],
                           "lap_end": [3, 6], "compound": ["MEDIUM", "HARD"], "tyre_age_at_start": [0, 0]})
    drivers = pd.DataFrame({"driver_number": [1], "name_acronym": ["AAA"], "team_name": ["T"]})
    lap_table, _ = lo.build_tables(laps, stints, drivers)
    assert lap_table["LapNumber"].is_unique and len(lap_table) == 6


def test_safety_car_reaction_saves_time_when_a_stop_is_due():
    L = 53
    d = sim.build_draws(B, L, np.random.default_rng(1), sc_laps=[18, 19, 20, 21])
    cfg = sim.SimConfig(react_to_sc=True)
    s = [(("MEDIUM", 26), ("HARD", 27))]
    fixed = sim.simulate_all(s, d, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=False).mean()
    reactive = sim.simulate_all(s, d, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=True).mean()
    assert fixed - reactive > 5.0  # a cheap stop under the safety car


def test_no_reaction_when_tyres_are_too_young_or_stop_is_far_away():
    L = 53
    cfg = sim.SimConfig(react_to_sc=True)
    young = sim.build_draws(B, L, np.random.default_rng(1), sc_laps=[4, 5, 6])
    s = [(("MEDIUM", 12), ("HARD", 41))]
    a = sim.simulate_all(s, young, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=False)
    b = sim.simulate_all(s, young, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=True)
    assert np.array_equal(a, b)
    far = sim.build_draws(B, L, np.random.default_rng(1), sc_laps=[10, 11, 12])
    s2 = [(("MEDIUM", 30), ("HARD", 23))]       # stop due 18+ laps after the safety car
    assert np.array_equal(sim.simulate_all(s2, far, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=False),
                          sim.simulate_all(s2, far, L, 83.0, -0.05, 23.0, cfg, caps=CAPS, react=True))


def test_reaction_respects_the_stint_limit_of_the_next_set():
    L = 53
    cfg = sim.SimConfig(react_to_sc=True)
    d = sim.build_draws(B, L, np.random.default_rng(1), sc_laps=[18, 19, 20, 21])
    s = [(("MEDIUM", 26), ("HARD", 27))]
    tight = {"SOFT": 25, "MEDIUM": 34, "HARD": 27}   # hard set cannot run longer than planned
    a = sim.simulate_all(s, d, L, 83.0, -0.05, 23.0, cfg, caps=tight, react=False)
    b = sim.simulate_all(s, d, L, 83.0, -0.05, 23.0, cfg, caps=tight, react=True)
    assert np.array_equal(a, b)


def test_safety_car_laps_are_detected_from_the_whole_field_being_slow():
    rows = []
    for drv in range(10):
        for lap in range(1, 58):
            t = 90.0 + 0.1 * ((drv + lap) % 3)
            if 20 <= lap <= 23:
                t = 130.0
            if lap == 40:
                t = 118.0          # a single slow lap is not a safety-car period
            rows.append(dict(Driver=drv, LapNumber=lap, LapTimeSec=t))
    laps, red = sim.detect_safety_car_laps(pd.DataFrame(rows), 90.0)
    assert laps == [20, 21, 22, 23] and red is False


def test_real_strategy_is_not_a_candidate():
    # the grid never contains the odd stop lap below, so a real strategy using it cannot be "found"
    strats = sim.enumerate_strategies(53, CAPS, step1=2)
    assert (("MEDIUM", 31), ("HARD", 22)) not in strats
