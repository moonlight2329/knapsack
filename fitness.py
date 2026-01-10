import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data/dataset.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def feature_selection_fitness(chromosome):
    selected_features = chromosome == 1

    if np.sum(selected_features) == 0:
        return 1e6  # Penalize empty feature set

    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(
        model,
        X.loc[:, selected_features],
        y,
        cv=5,
        scoring="accuracy"
    )

    accuracy = scores.mean()

    # Multi-objective: accuracy + feature reduction
    penalty = 0.01 * np.sum(selected_features)
    return -(accuracy - penalty)  # GA minimizes
