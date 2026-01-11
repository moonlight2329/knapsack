import streamlit as st
import matplotlib.pyplot as plt

from ga import GeneticAlgorithm
from fitness import load_instance, get_available_instances

st.title("Genetic Algorithm – MKnapCB3 Knapsack Optimization")

# ---- Auto-detect instances ----
instances = get_available_instances()

instance_id = st.selectbox(
    "Select Knapsack Instance",
    instances
)

values, weights, CAPACITY = load_instance(instance_id)

st.write("Number of items:", len(values))
st.write("Knapsack capacity:", CAPACITY)

# ---- GA Parameters ----
st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)

def knapsack_fitness(chromosome):
    total_value = (values * chromosome).sum()
    total_weight = (weights * chromosome).sum()

    if total_weight > CAPACITY:
        return 1e6

    return -total_value

# ---- Run GA ----
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

    st.success("Optimization Completed")
    st.write("Total Value:", -best_fitness)
    st.write("Total Weight:", (best_solution * weights).sum())

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("GA Convergence Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    st.pyplot(fig)
