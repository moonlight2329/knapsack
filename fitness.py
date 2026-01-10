import numpy as np
import os

# Locate dataset safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "knapsack.txt")

def load_knapsack(filepath):
    values, weights = [], []
    with open(filepath, "r") as f:
        for line in f:
            v, w = line.strip().split()
            values.append(float(v))
            weights.append(float(w))
    return np.array(values), np.array(weights)

values, weights = load_knapsack(DATA_FILE)

CAPACITY = 15  # Change if needed

def knapsack_fitness(chromosome):
    total_value = np.sum(values * chromosome)
    total_weight = np.sum(weights * chromosome)

    if total_weight > CAPACITY:
        return 1e6  # penalty

    return -total_value
