import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_mknapcb3", "mknapcb3")

def load_instance(instance_id):
    csv_file = os.path.join(DATA_DIR, f"mknapcb3_{instance_id}.csv")
    json_file = os.path.join(DATA_DIR, f"mknapcb3_{instance_id}_config.json")

    # Load CSV (items)
    df = pd.read_csv(csv_file)
    values = df.iloc[:, 0].astype(float).values
    weights = df.iloc[:, 1].astype(float).values

    # Load JSON (capacity)
    with open(json_file, "r") as f:
        config = json.load(f)

    capacity = config.get("capacity") or config.get("Capacity")

    return values, weights, float(capacity)
