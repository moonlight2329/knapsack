import numpy as np
import os

def load_knapsack_from_directory(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]

    if not files:
        raise FileNotFoundError("No .txt files found in knapsack directory")

    file_path = os.path.join(dir_path, files[0])  # load first instance

    values = []
    weights = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                values.append(float(parts[0]))
                weights.append(float(parts[1]))

    return np.array(values), np.array(weights), file_path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "mknapcb3")

values, weights, loaded_file = load_knapsack_from_directory(DATA_DIR)

CAPACITY = 15  # adjust if needed

def knapsack_fitness(chromosome):
    total_value = np.sum(values * chromosome)
    total_weight = np.sum(weights * chromosome)

    if total_weight > CAPACITY:
        return 1e6

    return -total_value
