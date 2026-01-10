import streamlit as st
import matplotlib.pyplot as plt

from ga import GeneticAlgorithm
from fitness import knapsack_fitness, values, weights

st.title("Genetic Algorithm – Knapsack Problem")

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)

if st.button("Run Genetic Algorithm"):
    ga = GeneticAlgorithm(
        fitness_func=knapsack_fitness,
        chromosome_length=len(values),
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    best_solution, best_fitness, history = ga.run()

    total_value = -best_fitness
    total_weight = (best_solution * weights).sum()

    st.success("Optimization Completed")
    st.write("Selected Items (1 = chosen):", best_solution)
    st.write("Total Value:", total_value)
    st.write("Total Weight:", total_weight)

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("GA Convergence Curve")
    st.pyplot(fig)
