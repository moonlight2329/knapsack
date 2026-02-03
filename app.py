import io
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_instance
from src.nsga2 import run_nsga2

st.set_page_config(page_title="NSGA II Group 6", layout="wide")
st.title("NSGA II for Multi Objective Knapsack")

DATASET_DIR = "dataset"

with st.sidebar:
    st.header("Select instance")
    inst = st.selectbox("Instance", ["inst01", "inst02", "inst03", "inst04", "inst05"], index=2)

    st.header("NSGA II parameters")
    pop_size = st.number_input("Population size", min_value=20, max_value=500, value=120, step=10)
    generations = st.number_input("Generations", min_value=20, max_value=2000, value=200, step=10)
    cx_prob = st.slider("Crossover probability", 0.0, 1.0, 0.9, 0.05)
    mut_prob = st.slider("Mutation probability per bit", 0.0, 0.2, 0.02, 0.005)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=1, step=1)

    run_btn = st.button("Run NSGA II", type="primary")

# Load data
try:
    items_df, meta = load_instance(DATASET_DIR, inst)
    cap_w1 = float(meta["capacity_w1"])
except Exception as e:
    st.error(f"Failed to load dataset files. {e}")
    st.stop()

st.caption(f"Instance: {inst} | Items: {len(items_df)} | capacity_w1: {cap_w1}")

if run_btn:
    with st.spinner("Running NSGA II..."):
        pareto_df, conv_df, summary = run_nsga2(
            items_df=items_df,
            cap_w1=cap_w1,
            pop_size=int(pop_size),
            generations=int(generations),
            cx_prob=float(cx_prob),
            mut_prob=float(mut_prob),
            seed=int(seed),
        )
    st.session_state["pareto_df"] = pareto_df
    st.session_state["conv_df"] = conv_df
    st.session_state["summary"] = summary

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    pareto_df = st.session_state["pareto_df"]
    conv_df = st.session_state["conv_df"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Final hypervolume", f"{summary['final_hypervolume']:.6f}")
    c2.metric("Pareto size", f"{summary['pareto_size']}")
    c3.metric("Runtime (s)", f"{summary['runtime_seconds']:.2f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Pareto front")
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(pareto_df["w2"], pareto_df["value"])
        plt.xlabel("Secondary weight (w2)")
        plt.ylabel("Total value")
        plt.title(f"Pareto Front NSGA II on {inst}")
        plt.grid(True)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Convergence")
        fig = plt.figure(figsize=(6, 4))
        plt.plot(conv_df["generation"], conv_df["hypervolume"])
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.title(f"Hypervolume Convergence NSGA II on {inst}")
        plt.grid(True)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Pareto points")
    st.dataframe(pareto_df, use_container_width=True)

    # download buttons
    st.subheader("Export")
    st.download_button("Download Pareto CSV", pareto_df.to_csv(index=False), file_name=f"{inst}_nsga2_pareto.csv")
    st.download_button("Download Convergence CSV", conv_df.to_csv(index=False), file_name=f"{inst}_nsga2_convergence.csv")
else:
    st.info("Select instance and parameters, then click Run NSGA II.")
