import numpy as np


def fit_line_total_least_squares(points):
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    covariance = centered_points.T @ centered_points
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    offset = -(normal @ centroid)
    return normal, offset


def line_through_two_points(first_point, second_point):
    direction = second_point - first_point
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    offset = -(normal @ first_point)
    return normal, offset


def distances_to_line(points, line):
    normal, offset = line
    return np.abs(points @ normal + offset)


def line_endpoints_for_plot(line):
    normal, offset = line
    point_on_line = -offset * normal
    direction = np.array([-normal[1], normal[0]])
    start = point_on_line - direction
    end = point_on_line + direction
    return start, end
