"""
Streamlit demo: pick a circuit, see ranked pit-stop strategies with uncertainty.

Run:    streamlit run app.py
Needs:  models/tyre_model_fe.json and models/circuit_profiles.json (see README, "Quick start").
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import simulate_strategy as sim

MODEL_PATH = Path("models/tyre_model_fe.json")
PROFILE_PATH = Path("models/circuit_profiles.json")
COLORS = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "tab:gray"}

st.set_page_config(page_title="F1 Strategy Engine", page_icon="🏎️", layout="wide")


@st.cache_data
def load_files():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["circuits"]
    return model, profiles


@st.cache_data(show_spinner="Simulating races...")
def analyze(circuit: str, n_sims: int, p_sc: float, pit_loss: float, lap_noise: float, react: bool):
    model, profiles = load_files()
    prof = profiles[circuit]
    cfg = sim.SimConfig(n_sims=n_sims, p_safety_car=p_sc, lap_noise_s=lap_noise, react_to_sc=react)
    B, _, trend = sim.load_model(str(MODEL_PATH))
    L, ref = prof["race_laps"], prof["typical_lap_s"]
    rng = np.random.default_rng(42)
    d = sim.build_draws(B, L, rng, cfg)
    strats = sim.enumerate_strategies(L, prof["stint_caps"])
    res = sim.simulate_all(strats, d, L, ref, trend, pit_loss, cfg, caps=prof["stint_caps"])
    mean = res.mean(axis=1)
    order = np.argsort(mean)
    best = order[0]
    gain = 0.0
    if react:  # what the best plan would lose if it could not react to the safety car
        fixed = sim.simulate_all([strats[best]], d, L, ref, trend, pit_loss, cfg, caps=prof["stint_caps"], react=False)
        gain = float(fixed.mean() - mean[best])
    rows = []
    for rank, i in enumerate(order[:12], 1):
        delta = res[i] - res[best]
        lo, hi = np.percentile(delta, [10, 90])
        rows.append({
            "Rank": rank, "Strategy": sim.label(strats[i]), "Stops": len(strats[i]) - 1,
            "Avg vs best (s)": round(float(mean[i] - mean[best]), 1),
            "10% (s)": round(float(lo), 1), "90% (s)": round(float(hi), 1),
            "Beats #1": f"{100 * (res[i] < res[best]).mean():.0f}%",
        })
    best_by_stops = {}
    for n_st in (1, 2):
        idxs = [i for i, s in enumerate(strats) if len(s) == n_st + 1]
        if idxs:
            j = min(idxs, key=lambda i: mean[i])
            best_by_stops[n_st] = (sim.label(strats[j]), round(float(mean[j] - mean[best]), 1))
    return pd.DataFrame(rows), len(strats), best_by_stops, gain


def tyre_chart(model):
    B = np.array(model["boot_betas"])
    feats = model["features"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in sim.COMPOUNDS:
        a, b = feats.index(f"{c}_age"), feats.index(f"{c}_age2")
        x = np.linspace(0, model["compounds"][c]["max_age_p95"], 50)
        curves = B[:, [a]] * x + B[:, [b]] * x ** 2
        ax.plot(x, np.median(curves, axis=0), color=COLORS[c], lw=2, label=c)
        ax.fill_between(x, np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0),
                        color=COLORS[c], alpha=0.18)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Time lost vs fresh tyre (s/lap)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def strategy_chart(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    top = df.head(10).iloc[::-1]
    mean = top["Avg vs best (s)"].to_numpy()
    lo, hi = top["10% (s)"].to_numpy(), top["90% (s)"].to_numpy()
    ax.barh(top["Strategy"], mean, xerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)],
            color="tab:blue", alpha=0.75, capsize=3)
    ax.set_xlabel("Seconds slower than #1")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


if not (MODEL_PATH.exists() and PROFILE_PATH.exists()):
    st.title("🏎️ F1 Strategy Engine")
    st.error("Model files not found. Run the pipeline first: `python fit_degradation_fe.py` and "
             "`python build_circuit_profiles.py` (see the README, Quick start).")
    st.stop()

model, profiles = load_files()
circuits = sorted(profiles)

with st.sidebar:
    st.header("Settings")
    circuit = st.selectbox("Circuit", circuits, index=circuits.index("Monza") if "Monza" in circuits else 0)
    prof = profiles[circuit]
    n_sims = st.slider("Simulated races per strategy", 100, 1000, 400, 100)
    p_sc = st.slider("Chance of a safety car", 0.0, 1.0, 0.5, 0.05)
    pit_loss = st.slider("Pit-stop time loss (s)", 12.0, 35.0, float(round(prof["pit_loss_s"], 1)), 0.5,
                         key=f"pit_{circuit}")
    lap_noise = st.slider("Lap-time noise (s)", 0.1, 1.0, 0.4, 0.05)
    react = st.checkbox("Pit under a safety car if a stop is due soon", value=True,
                        help="If a safety car comes out when the tyres have done 8+ laps and the planned "
                             "stop is within 12 laps, the stop is pulled forward and costs far less.")
    st.caption("Defaults for pit loss come from real stops at this circuit.")

st.title("🏎️ F1 Strategy Engine")
st.caption("Monte Carlo pit-stop strategy simulator built on a tyre-degradation model fitted to "
           "2023-2025 race data. Unofficial research project.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Race laps", prof["race_laps"])
c2.metric("Typical lap", f"{prof['typical_lap_s']:.1f} s")
c3.metric("Pit-stop loss", f"{pit_loss:.1f} s")
caps = prof["stint_caps"]
c4.metric("Longest stint (S / M / H)", f"{caps['SOFT']} / {caps['MEDIUM']} / {caps['HARD']}")

df, n_strats, best_by_stops, gain = analyze(circuit, n_sims, p_sc, pit_loss, lap_noise, react)

tab1, tab2, tab3, tab4 = st.tabs(["Ranked strategies", "Tyre model", "What teams ran", "About"])

with tab1:
    best = df.iloc[0]
    st.subheader(f"Best average strategy: {best['Strategy']}")
    st.caption(f"{n_strats} strategies x {n_sims} simulated races. Letters are compounds (S/M/H) "
               "and numbers are laps on that set. Strategies within a second or two of each other "
               "are effectively tied.")
    left, right = st.columns([5, 4])
    left.dataframe(df, hide_index=True,
                   column_config={"Strategy": st.column_config.TextColumn("Strategy", width="medium")})
    right.pyplot(strategy_chart(df))
    right.caption("Bars: average gap to #1. Whiskers: 10-90% of simulated races.")
    notes = [f"Best {n}-stop: **{lab}** ({d:+.1f} s vs overall best)" for n, (lab, d) in best_by_stops.items()]
    if react:
        notes.append(f"Reacting to safety cars is worth about **{gain:.1f} s** on average to the best strategy.")
    st.write("  \n".join(notes))

with tab2:
    st.write("Time lost per lap as a tyre ages, after removing fuel burn, track evolution, driver pace "
             "and laps spent in dirty air. Bands are 90% bootstrap intervals over races.")
    col_a, col_b = st.columns([3, 2])
    col_a.pyplot(tyre_chart(model))
    off = model["pace_offset_vs_soft"]
    col_b.markdown("**Fresh-tyre pace vs soft (s/lap, slower is positive)**")
    col_b.table(pd.DataFrame({"Compound": list(off), "Gap": [f"{v:+.2f}" for v in off.values()]}))
    col_b.caption("Medium and hard are hard to tell apart in this data. Treat that choice as uncertain.")

with tab3:
    act = prof.get("actual")
    if not act:
        st.info("No dry race with usable data for this circuit.")
    else:
        st.write(f"What drivers ran in the {act['year']} race:")
        a1, a2 = st.columns(2)
        a1.markdown("**Number of stops**")
        a1.table(pd.DataFrame({"Stops": list(act["stops"]), "Drivers": list(act["stops"].values())}))
        a2.markdown("**Most common compound sequences**")
        a2.table(pd.DataFrame({"Sequence": list(act["sequences"]), "Drivers": list(act["sequences"].values())}))
        st.caption("S = soft, M = medium, H = hard.")

with tab4:
    st.markdown(
        """
**How it works.** A two-way fixed-effects regression learns how much each compound slows down as it
ages, comparing drivers on the same lap of the same race. A Monte Carlo simulator then races every
1-stop and 2-stop strategy hundreds of times, drawing a different fitted tyre model, lap noise, pit-stop
time and a possible safety car each time, and ranks strategies by average race time. Optionally, a plan
pulls a due stop forward when a safety car comes out.

**Limits.** One car alone on track (no traffic or opponents), one tyre model for all circuits, a simple
safety-car model, and stint-length limits learned from past seasons. Medium vs hard is poorly separated
by the data, so treat that choice as uncertain. See the README for how the model was back-tested.

Data from the free [OpenF1](https://openf1.org) API. Not affiliated with Formula 1 or any team.
        """
    )
