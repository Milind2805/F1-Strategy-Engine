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

Back-tested on the 20 dry 2025 races (wet and chaotic races are skipped). For each race it compares
the simulator's best strategy with the strategy most finishers actually ran. The tyre model in the last
row was fitted **without** 2025, so that row is a clean out-of-sample test.

| Version | Stop count matches majority | Compound set matches | Median regret of the real strategy | Races under 5 s / 10 s | Real strategy's rank among simulated options |
|---|---|---|---|---|---|
| v1: global stint limits | 62% | 0% | 4.3 s | 52% / 67% | top 5% |
| v2: stint limits and pit loss learned from other seasons | 80% | 35% | 2.6 s | 60% / 80% | top 2% |
| v2, tyre model fitted without 2025 | 80% | 35% | 2.3 s | 65% / 80% | top 2% |

"Regret" is how much slower, in simulation, the strategy teams really ran is than the simulator's best.
v1 covers 21 races (one more than v2, which skips a chaotic race).

Tyre model, on unseen races, after removing driver and race-lap effects: lap-time error fell from
0.633 s to 0.529 s (16.5% lower) once the tyre terms were added.

### Read these numbers carefully

- **Low regret is weaker evidence than it looks.** The model sees many strategies as nearly tied, so
  almost anything sits close to the best.
- **Medium and hard are poorly separated** by the data. Soft wears clearly faster; medium vs hard is
  uncertain. The compound-set match (35%) reflects this.
- **Stint limits do real work.** They come from how long teams ran each compound at that circuit in
  other seasons, so part of the agreement is learned team behaviour, not tyre physics alone.
- **The biggest misses are races where teams reacted mid-race** (safety cars, red flags, tyre-life
  rules). The simulator plans each strategy in advance and cannot react yet.

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
   hundreds of times on shared random draws, so differences between strategies are fair.
4. **Backtest** (`backtest_strategy.py`): compares the simulator with real race strategies, using only
   data from other seasons for pit loss and stint limits.
5. **Demo** (`app.py`): Streamlit app.

## Limitations

- One car alone on track: no traffic, opponents or undercuts.
- One tyre model for all circuits (a circuit-specific version gained only 2.2%, so it is not used).
- Simple safety-car model with no mid-race reactions.
- Fixed pre-planned strategies only, with at most 2 stops considered when ranking.
- Dry races only.
- Soft-tyre drop-off ("cliff") is not modelled; teams pit before it, so the data cannot show it.

## Data and licence

Code is MIT licensed (see `LICENSE`). Lap and stint data comes from the OpenF1 API; check its terms
before redistributing the data. This repository does not include the raw data.
