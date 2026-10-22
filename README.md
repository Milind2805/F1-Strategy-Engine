# F1 Strategy Engine

A Monte Carlo pit-stop strategy simulator for Formula 1, built on a tyre-degradation model fitted to
2023-2025 race data and back-tested against what teams actually ran.

> Unofficial research project. Not affiliated with Formula 1 or any team. Data from the free
> [OpenF1](https://openf1.org) API.

<!-- Add a screenshot of the demo here: ![demo](docs/demo.png) -->
<!-- Add the tyre chart here: ![tyre model](docs/tyre_degradation_fe.png) -->

## What it does

Pick a circuit and it ranks 1-stop and 2-stop strategies by expected race time, with the uncertainty
around each one. Every simulated race draws a different fitted tyre model, lap-to-lap noise, pit-stop
time and a possible safety car, so the answer is a range, not a single number. Strategies that are
within a second or two of each other are effectively tied, and the tool says so.

## Results

**Tyre model.** On unseen races, after removing driver and race-lap effects, lap-time error fell from
0.633 s to 0.529 s (16.5% lower) once the tyre terms were added. Soft wears clearly faster than hard;
medium vs hard cannot be told apart with this data.

**Strategy back-test.** Back-test numbers are being re-measured. An earlier version of the back-test
added the real strategy to the simulator's own candidate list, so whenever the real strategy beat every
other candidate the simulator "agreed" with it by construction. That flattered the agreement figures,
and they have been removed. The corrected back-test simulates the real strategy separately and reports
regret that can be negative. Reproduce it with:

```bash
python fit_degradation_fe.py 2025                       # tyre model fitted without 2025
python backtest_strategy.py 2025 --holdout              # random safety car, plans fixed in advance
python backtest_strategy.py 2025 --holdout --replay     # real safety-car laps, plans may pit under them
```

Results will be added here, including the original (flawed) numbers for comparison, once re-run.

### How to read the back-test

- **Low regret is weaker evidence than it looks** while the model sees many strategies as nearly tied.
- **Medium and hard are poorly separated**, which limits how well compounds can be matched.
- **Stint limits do real work.** They come from how long teams ran each compound at that circuit in
  other seasons, so part of any agreement is learned team behaviour, not tyre physics alone.
- **Races where teams reacted mid-race** (safety cars, red flags, tyre-life rules) are hard for a
  plan-in-advance simulator. The replay mode (`--replay`) feeds it the real safety-car laps.

## How it works

1. **Data** (`load_openf1.py`, `load_season.py`): lap times, tyre compounds and stints from OpenF1.
   Each lap gets a "clean" flag (green flag, not an in-lap or out-lap, not a slow outlier) and the gap
   to the car ahead, so laps spent in dirty air can be excluded.
2. **Tyre model** (`fit_degradation_fe.py`): a two-way fixed-effects regression. Every race-lap has its
   own effect (fuel burn, track evolution, safety cars, weather) and every driver-race has its own pace,
   so tyre wear is learned only from drivers on the same lap running different tyres or ages. A
   race-level bootstrap gives confidence intervals. Earlier versions (`fit_degradation.py`,
   `fit_degradation_by_circuit.py`) are kept as baselines.
3. **Simulator** (`simulate_strategy.py`): enumerates strategies within stint limits and races each one
   hundreds of times on shared random draws, so differences between strategies are fair. Plans can pit
   under a safety car when a stop is due soon (on by default in the demo).
4. **Backtest** (`backtest_strategy.py`): compares the simulator with real race strategies, using only
   data from other seasons for pit loss and stint limits. The real strategy is never one of the candidates.
5. **Demo** (`app.py`): Streamlit app.

## Quick start

Python 3.10+.

```bash
pip install -r requirements.txt

# 1. download seasons (about 5 minutes each; the free API is rate limited)
python load_season.py 2023
python load_season.py 2024
python load_season.py 2025

# 2. fit the tyre model and summarise circuits (these two files are what the demo needs)
python fit_degradation_fe.py
python build_circuit_profiles.py

# 3. try it
python simulate_strategy.py Monza 2025
streamlit run app.py

# 4. back-test (clean out-of-sample version)
python fit_degradation_fe.py 2025
python backtest_strategy.py 2025 --holdout
python backtest_strategy.py 2025 --holdout --replay
```

Commit `models/` (small JSON files) so the demo runs without the raw data. To deploy free, push to
GitHub and point [Streamlit Community Cloud](https://streamlit.io/cloud) at `app.py`.

Run the tests with `pip install -r requirements-dev.txt` then `pytest`.

## Limitations

- One car alone on track: no traffic, opponents or undercuts.
- One tyre model for all circuits (a circuit-specific version gained only 2.2%, so it is not used).
- Simple safety-car model. Plans can pit under a safety car, but there are no other mid-race decisions.
- Fixed pre-planned strategies only, with at most 2 stops considered when ranking.
- Dry races only.
- Soft-tyre drop-off ("cliff") is not modelled; teams pit before it, so the data cannot show it.

## Roadmap

- [x] Safety-car reactions (pit under a safety car if a stop is due soon)
- [ ] Tune and validate the reaction rule; virtual safety cars
- [ ] Model the soft-tyre cliff
- [ ] A racing game environment where a reinforcement-learning agent learns to drive, with this engine
      as the strategy agent
- [ ] Setup optimisation for Formula Student and sim-racing users

## Data and licence

Code is MIT licensed (see `LICENSE`). Lap and stint data comes from the OpenF1 API; check its terms
before redistributing the data. This repository does not include the raw data.
