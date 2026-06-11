import numpy as np

from line_model import fit_line_total_least_squares, distances_to_line


def fit_two_lines(points):
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


def run_alternating_fit(points, assignment):
    for iteration in range(50):
        lines = fit_one_line_per_group(points, assignment)
        if lines is None:
            return None, np.inf
        new_assignment = assign_each_point_to_nearest_line(points, lines)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
    cost = total_squared_distance(points, lines)
    return lines, cost


def fit_one_line_per_group(points, assignment):
    lines = []
    for group_label in (0, 1):
        group_points = points[assignment == group_label]
        if len(group_points) < 2:
            return None
        line = fit_line_total_least_squares(group_points)
        lines.append(line)
    return lines


def assign_each_point_to_nearest_line(points, lines):
    distances_to_first = distances_to_line(points, lines[0])
    distances_to_second = distances_to_line(points, lines[1])
    assignment = (distances_to_second < distances_to_first).astype(int)
    return assignment


def total_squared_distance(points, lines):
    distances_to_first = distances_to_line(points, lines[0])
    distances_to_second = distances_to_line(points, lines[1])
    nearest_distance = np.minimum(distances_to_first, distances_to_second)
    return float(np.sum(nearest_distance ** 2))
