import numpy as np

def load_knapsack_data(filepath):
    values = []
    weights = []

    with open(filepath, "r") as f:
        for line in f:
            v, w = line.strip().split()
            values.append(float(v))
            weights.append(float(w))

    return np.array(values), np.array(weights)

# Load dataset
values, weights = load_knapsack_data("data/mknapcb3.txt")

CAPACITY = 15  # Change based on dataset

def knapsack_fitness(chromosome):
    total_value = np.sum(values * chromosome)
    total_weight = np.sum(weights * chromosome)

    if total_weight > CAPACITY:
        return 1e6  # Penalize infeasible solutions

    return -total_value  # Maximize value
