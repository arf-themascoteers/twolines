import numpy as np


def sample_points_near_line(center, angle, rng):
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    points = []
    while len(points) < 50:
        position_along_line = rng.uniform(-0.7, 0.7)
        perpendicular_noise = rng.normal(0.0, 0.02)
        point = center + position_along_line * direction + perpendicular_noise * normal
        point_is_inside_square = np.all(np.abs(point) <= 0.5)
        if point_is_inside_square:
            points.append(point)
    return np.array(points)


def combine_two_point_clouds(first_points, second_points, rng):
    all_points = np.vstack([first_points, second_points])
    rng.shuffle(all_points)
    return all_points


def make_parallel_case(rng):
    angle = rng.uniform(0.0, np.pi)
    normal = np.array([-np.sin(angle), np.cos(angle)])
    first_points = sample_points_near_line(0.2 * normal, angle, rng)
    second_points = sample_points_near_line(-0.2 * normal, angle, rng)
    return combine_two_point_clouds(first_points, second_points, rng)


def make_perpendicular_case(rng):
    angle = rng.uniform(0.0, np.pi)
    first_points = sample_points_near_line(np.zeros(2), angle, rng)
    second_points = sample_points_near_line(np.zeros(2), angle + np.pi / 2, rng)
    return combine_two_point_clouds(first_points, second_points, rng)


def make_random_case(rng):
    first_center = rng.uniform(-0.25, 0.25, size=2)
    second_center = rng.uniform(-0.25, 0.25, size=2)
    first_angle = rng.uniform(0.0, np.pi)
    second_angle = rng.uniform(0.0, np.pi)
    first_points = sample_points_near_line(first_center, first_angle, rng)
    second_points = sample_points_near_line(second_center, second_angle, rng)
    return combine_two_point_clouds(first_points, second_points, rng)


def build_all_cases():
    rng = np.random.default_rng(42)
    cases = []
    cases.append(("Parallel", make_parallel_case(rng)))
    cases.append(("Perpendicular", make_perpendicular_case(rng)))
    cases.append(("Random 1", make_random_case(rng)))
    cases.append(("Random 2", make_random_case(rng)))
    cases.append(("Random 3", make_random_case(rng)))
    return cases
