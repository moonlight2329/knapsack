import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class Individual:
    x: np.ndarray
    value: float
    w1: float
    w2: float
    feasible: bool
    rank: int = 10**9
    crowding: float = 0.0


def evaluate_population(X, values, w1s, w2s, cap_w1) -> List[Individual]:
    total_value = X @ values
    total_w1 = X @ w1s
    total_w2 = X @ w2s
    feasible = total_w1 <= cap_w1

    pop: List[Individual] = []
    for i in range(X.shape[0]):
        pop.append(
            Individual(
                x=X[i].copy(),
                value=float(total_value[i]),
                w1=float(total_w1[i]),
                w2=float(total_w2[i]),
                feasible=bool(feasible[i]),
            )
        )
    return pop


def dominates(a: Individual, b: Individual) -> bool:
    if a.feasible and (not b.feasible):
        return True
    if (not a.feasible) and b.feasible:
        return False

    if (not a.feasible) and (not b.feasible):
        return a.w1 < b.w1

    not_worse = (a.value >= b.value) and (a.w2 <= b.w2)
    strictly_better = (a.value > b.value) or (a.w2 < b.w2)
    return not_worse and strictly_better


def fast_nondominated_sort(pop: List[Individual]) -> List[List[int]]:
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()
    return fronts


def crowding_distance(pop: List[Individual], front: List[int]) -> None:
    if not front:
        return

    for idx in front:
        pop[idx].crowding = 0.0

    obj1 = np.array([pop[i].value for i in front], dtype=float)
    obj2 = np.array([pop[i].w2 for i in front], dtype=float)

    def assign(sorted_idx, obj_vals):
        pop[front[sorted_idx[0]]].crowding = float("inf")
        pop[front[sorted_idx[-1]]].crowding = float("inf")
        vmin = obj_vals[sorted_idx[0]]
        vmax = obj_vals[sorted_idx[-1]]
        if vmax - vmin == 0:
            return
        for k in range(1, len(sorted_idx) - 1):
            mid = front[sorted_idx[k]]
            prev_v = obj_vals[sorted_idx[k - 1]]
            next_v = obj_vals[sorted_idx[k + 1]]
            pop[mid].crowding += (next_v - prev_v) / (vmax - vmin)

    assign(np.argsort(obj1), obj1)
    assign(np.argsort(obj2), obj2)


def tournament_select(pop: List[Individual], k: int, rng: np.random.Generator) -> Individual:
    cand = rng.choice(len(pop), size=k, replace=False)
    best = cand[0]
    for i in cand[1:]:
        if pop[i].rank < pop[best].rank:
            best = i
        elif pop[i].rank == pop[best].rank and pop[i].crowding > pop[best].crowding:
            best = i
    return pop[best]


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    mask = rng.random(a.shape[0]) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2


def bitflip_mutation(x: np.ndarray, pm: float, rng: np.random.Generator):
    m = rng.random(x.shape[0]) < pm
    y = x.copy()
    y[m] = 1 - y[m]
    return y


def get_pareto_front(pop: List[Individual]) -> List[Individual]:
    feas = [ind for ind in pop if ind.feasible]
    if not feas:
        return []
    nd = []
    for a in feas:
        dominated_flag = False
        for b in feas:
            if b is a:
                continue
            if dominates(b, a):
                dominated_flag = True
                break
        if not dominated_flag:
            nd.append(a)
    return nd


def hypervolume_2d(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    pts = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]
    hv = 0.0
    best_y = 0.0
    for x, y in pts:
        if y > best_y:
            hv += x * (y - best_y)
            best_y = y
    return float(hv)


def compute_hv(pop: List[Individual], v_sum: float, w2_sum: float) -> float:
    nd = get_pareto_front(pop)
    if not nd:
        return 0.0

    vals = np.array([p.value for p in nd], dtype=float)
    w2s = np.array([p.w2 for p in nd], dtype=float)

    v_norm = np.clip(vals / (v_sum if v_sum != 0 else 1.0), 0, 1)
    w_norm = np.clip(w2s / (w2_sum if w2_sum != 0 else 1.0), 0, 1)

    benefit = np.column_stack([v_norm, 1.0 - w_norm])
    return hypervolume_2d(benefit)


def run_nsga2(
    items_df: pd.DataFrame,
    cap_w1: float,
    pop_size: int,
    generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
    tour_k: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:

    rng = np.random.default_rng(seed)

    values = items_df["value"].to_numpy(dtype=float)
    w1s = items_df["w1"].to_numpy(dtype=float)
    w2s = items_df["w2"].to_numpy(dtype=float)
    n_items = len(items_df)

    v_sum = float(values.sum())
    w2_sum = float(w2s.sum())

    X = (rng.random((pop_size, n_items)) < 0.5).astype(np.int8)
    pop = evaluate_population(X, values, w1s, w2s, cap_w1)

    conv_rows = []
    t0 = time.time()

    for gen in range(generations):
        fronts = fast_nondominated_sort(pop)
        for f in fronts:
            crowding_distance(pop, f)

        # offspring
        offspring_X = []
        while len(offspring_X) < pop_size:
            p1 = tournament_select(pop, tour_k, rng)
            p2 = tournament_select(pop, tour_k, rng)
            c1, c2 = p1.x.copy(), p2.x.copy()

            if rng.random() < cx_prob:
                c1, c2 = uniform_crossover(c1, c2, rng)

            c1 = bitflip_mutation(c1, mut_prob, rng)
            c2 = bitflip_mutation(c2, mut_prob, rng)

            offspring_X.append(c1)
            if len(offspring_X) < pop_size:
                offspring_X.append(c2)

        offspring = evaluate_population(np.array(offspring_X, dtype=np.int8), values, w1s, w2s, cap_w1)

        combined = pop + offspring
        fronts = fast_nondominated_sort(combined)

        new_pop = []
        for f in fronts:
            crowding_distance(combined, f)
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend([combined[i] for i in f])
            else:
                sorted_f = sorted(f, key=lambda i: combined[i].crowding, reverse=True)
                need = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in sorted_f[:need]])
                break

        pop = new_pop

        hv = compute_hv(pop, v_sum, w2_sum)
        conv_rows.append({"generation": gen, "hypervolume": hv})

    nd = get_pareto_front(pop)
    pareto_df = pd.DataFrame(
        [{"value": p.value, "w2": p.w2, "w1": p.w1} for p in nd]
    ).sort_values(["w2", "value"], ascending=[True, False], ignore_index=True)

    conv_df = pd.DataFrame(conv_rows)

    summary = {
        "algorithm": "NSGA II",
        "seed": seed,
        "pop_size": pop_size,
        "generations": generations,
        "cx_prob": cx_prob,
        "mut_prob": mut_prob,
        "runtime_seconds": float(time.time() - t0),
        "pareto_size": int(len(pareto_df)),
        "final_hypervolume": float(conv_df["hypervolume"].iloc[-1]) if len(conv_df) else 0.0
    }

    return pareto_df, conv_df, summary
