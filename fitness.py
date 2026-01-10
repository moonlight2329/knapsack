import numpy as np
import os

def load_knapsack_data(filepath):
    values = []
    weights = []

    with open(filepath, "r") as f:
        for line in f:
            v, w = line.strip().split()
            values.append(float(v))
            weights.append(float(w))

    return np.array(values), np.array(weights)

# 🔥 FIXED PATH HANDLING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mknapcb3.txt")

values, weights = load_knapsack_data(DATA_PATH)

CAPACITY = 15

def knapsack_fitness(chromosome):
    total_value = np.sum(values * chromosome)
    total_weight = np.sum(weights * chromosome)

    if total_weight > CAPACITY:
        return 1e6

    return -total_value
