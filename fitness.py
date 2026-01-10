import numpy as np
import os

# 🔹 Automatically find first knapsack file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_mknapcb3", "mknapcb3")

def load_knapsack_instance(directory):
    files = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])
    if not files:
        raise FileNotFoundError("No knapsack .txt files found")

    file_path = os.path.join(directory, files[0])  # load first instance

    values, weights = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                values.append(float(parts[0]))
                weights.append(float(parts[1]))

    return np.array(values), np.array(weights), file_path

values, weights, LOADED_FILE = load_knapsack_instance(DATA_DIR)

CAPACITY = 15  # 🔴 change if dataset specifies otherwise

def knapsack_fitness(chromosome):
    total_value = np.sum(values * chromosome)
    total_weight = np.sum(weights * chromosome)

    if total_weight > CAPACITY:
        return 1e6

    return -total_value
