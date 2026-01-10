import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from ga import GeneticAlgorithm
from fitness import feature_selection_fitness, X

st.title("Genetic Algorithm – Feature Selection")

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 10, 200, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)

if st.button("Run GA"):
    ga = GeneticAlgorithm(
        fitness_func=feature_selection_fitness,
        chromosome_length=X.shape[1],
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    best_solution, best_fitness, history = ga.run()

    selected_features = X.columns[best_solution == 1]

    st.success("Optimization Complete")
    st.write("Selected Features:", list(selected_features))
    st.write("Best Fitness:", best_fitness)

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("GA Convergence Curve")
    st.pyplot(fig)
