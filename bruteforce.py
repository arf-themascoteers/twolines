import numpy as np

from alternating_fit import run_alternating_fit


def fit_bruteforce(points):
    restart_rng = np.random.default_rng(0)
    best_lines = None
    best_cost = np.inf
    for restart in range(30):
        initial_assignment = restart_rng.integers(0, 2, size=len(points))
        lines, cost = run_alternating_fit(points, initial_assignment)
        if lines is not None and cost < best_cost:
            best_cost = cost
            best_lines = lines
    return best_lines
