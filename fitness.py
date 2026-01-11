import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_mknapcb3", "mknapcb3")

def get_available_instances():
    instances = []
    for file in os.listdir(DATA_DIR):
        if file.endswith("_config.json"):
            instance_id = file.replace("mknapcb3_", "").replace("_config.json", "")
            instances.append(instance_id)
    return sorted(instances)

def extract_capacity(config):
    # Common capacity key names
    for key in ["capacity", "Capacity", "knapsack_capacity", "C"]:
        if key in config:
            return config[key]

    # Sometimes nested
    for value in config.values():
        if isinstance(value, dict):
            for key in ["capacity", "Capacity", "knapsack_capacity", "C"]:
                if key in value:
                    return value[key]

    raise KeyError(
        f"Capacity not found in config file. Available keys: {list(config.keys())}"
    )

def load_instance(instance_id):
    csv_file = os.path.join(DATA_DIR, f"mknapcb3_{instance_id}.csv")
    json_file = os.path.join(DATA_DIR, f"mknapcb3_{instance_id}_config.json")

    # Load CSV
    df = pd.read_csv(csv_file)
    values = df.iloc[:, 0].astype(float).values
    weights = df.iloc[:, 1].astype(float).values

    # Load JSON
    with open(json_file, "r") as f:
        config = json.load(f)

    capacity = extract_capacity(config)

    return values, weights, float(capacity)
