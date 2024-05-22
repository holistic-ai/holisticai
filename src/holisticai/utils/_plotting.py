# Base Imports
import numpy as np


def get_colors(n_lines, extended_colors=False, reverse=False):
    """
    Get colors for Holistic AI pallette

    Description
    ----------
    This function returns an array of colors
    for plotting using the Holistic AI pallette.

    Parameters
    ----------
    n_lines : scalar
        Number of lines

    Returns
    -------
    List of colors in RGB format
    """

    purple = np.array([161 / 255, 103 / 255, 211 / 255])
    blue = np.array([68 / 255, 168 / 255, 249 / 255])

    colors = [purple + t * (blue - purple) for t in np.linspace(0, 1, n_lines)]

    if extended_colors and n_lines > 3:
        n_lines_extremes = round(n_lines / 3) + 1
        n_lines_middle = n_lines - 2 * n_lines_extremes + 2

        purple_to_blue = [
            purple + t * (blue - purple) for t in np.linspace(0, 1, n_lines_middle)
        ]

        white = np.ones(blue.shape) * 0.8
        black = np.zeros(blue.shape) + 0.3

        black_to_purple = [
            black + t * (purple - black) for t in np.linspace(0, 1, n_lines_extremes)
        ]
        blue_to_white = [
            blue + t * (white - blue) for t in np.linspace(0, 1, n_lines_extremes)
        ]

        colors = black_to_purple[:-1] + purple_to_blue + blue_to_white[1:]

    if reverse:
        colors = colors[::-1]

    return colors
