import numpy as np

from line_model import fit_line_total_least_squares, distances_to_line, line_through_two_points


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


def fit_singleshot(points):
    rng = np.random.default_rng(0)
    initial_assignment = rng.integers(0, 2, size=len(points))
    lines, cost = run_alternating_fit(points, initial_assignment)
    return lines


def fit_besteffort(points):
    first_line = find_strongest_line_with_ransac(points)
    distance_from_first = distances_to_line(points, first_line)
    points_far_from_first_line = points[distance_from_first > 0.05]
    second_line = find_strongest_line_with_ransac(points_far_from_first_line)
    initial_assignment = assign_each_point_to_nearest_line(points, [first_line, second_line])
    lines, cost = run_alternating_fit(points, initial_assignment)
    return lines


def find_strongest_line_with_ransac(points):
    rng = np.random.default_rng(0)
    best_inlier_mask = None
    best_inlier_count = 0
    for trial in range(200):
        pair_indices = rng.choice(len(points), size=2, replace=False)
        candidate_line = line_through_two_points(points[pair_indices[0]], points[pair_indices[1]])
        distances = distances_to_line(points, candidate_line)
        inlier_mask = distances < 0.05
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
    return fit_line_total_least_squares(points[best_inlier_mask])


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
