import numpy as np

from alternating_fit import run_alternating_fit


def fit_singleshot(points):
    rng = np.random.default_rng(0)
    initial_assignment = rng.integers(0, 2, size=len(points))
    lines, cost = run_alternating_fit(points, initial_assignment)
    return lines
