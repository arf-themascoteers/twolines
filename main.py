import matplotlib.pyplot as plt
import numpy as np
from data_cases import build_all_cases
from line_model import line_endpoints_for_plot
from bruteforce import fit_bruteforce
from singleshot import fit_singleshot
from besteffort import fit_besteffort

fit_two_lines = fit_besteffort


def draw_points(axis, points, title):
    axis.scatter(points[:, 0], points[:, 1], s=8, color="gray")
    axis.set_xlim(-0.55, 0.55)
    axis.set_ylim(-0.55, 0.55)
    axis.set_aspect("equal")
    axis.set_title(title)


def draw_fitted_lines(axis, lines):
    for line, color in zip(lines, ("tab:red", "tab:blue")):
        start, end = line_endpoints_for_plot(line)
        axis.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)


def main():
    cases = build_all_cases()
    figure, axes = plt.subplots(2, len(cases), figsize=(4 * len(cases), 8))
    for column, (title, points) in enumerate(cases):
        draw_points(axes[0, column], points, title)
        draw_points(axes[1, column], points, title + " (fitted)")
        lines = fit_two_lines(points)
        draw_fitted_lines(axes[1, column], lines)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    main()
