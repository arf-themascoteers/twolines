import numpy as np

from alternating_fit import run_alternating_fit, assign_each_point_to_nearest_line
from line_model import fit_line_total_least_squares, distances_to_line, line_through_two_points


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
