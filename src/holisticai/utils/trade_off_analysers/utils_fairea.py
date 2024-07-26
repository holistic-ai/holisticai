import numpy as np


def get_closer_bounds(point, x_arr, y_arr, bound="lower"):
    """
    Get the closest point to the point in the array.

    Parameters
    ----------
    point : tuple
        The point to be compared.
    x_arr : array, shape = (n_samples,)
        The array of the x axis.
    y_arr : array, shape = (n_samples,)
        The array of the y axis.
    bound : string
        The bound to be compared if lower or upper.

    Returns
    -------
    x_point : float
        The closest point in the x axis.
    y_point : float
        The closest point in the y axis.
    """
    if point in x_arr:
        return point, y_arr[np.where(x_arr == point)[0][0]]
    bounds = x_arr[point > x_arr] if bound == "lower" else x_arr[point < x_arr]
    if np.any(bounds) == 0:
        return 0, 0
    x_point = bounds[np.argmin((bounds - point) ** 2)]
    return x_point, y_arr[np.where(x_arr == x_point)[0][0]]


def get_rect_equation(point_1, point_2):
    """
    Get the equation of the line segment.

    Parameters
    ----------
    point_1 : tuple
        The first point of the line segment.
    point_2 : tuple
        The second point of the line segment.

    Returns
    -------
    m : float
        The slope of the line segment.
    b : float
        The intercept of the line segment.
    """

    x1, y1 = point_1
    x2, y2 = point_2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def do_line_segments_intersect(segment1, segment2):
    """
    Check if two line segments intersect.

    Parameters
    ----------
    p1 : tuple
        The first point of the first line segment.
    q1 : tuple
        The second point of the first line segment.
    p2 : tuple
        The first point of the second line segment.
    q2 : tuple
        The second point of the second line segment.

    Returns
    -------
    bool: True if the line segments intersect, False otherwise.
    """

    def orientation(p, q, r):
        """Find the orientation of the triplet (p, q, r).

        Parameters:
        ----------
            p (tuple): The first point.
            q (tuple): The second point.
            r (tuple): The third point.

        Returns:
            int: 0 if p, q, and r are collinear, 1 if clockwise, and 2 if counterclockwise.
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    p1, q1 = segment1
    p2, q2 = segment2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if (
        o1 == 0
        and p2[0] <= max(p1[0], q1[0])
        and p2[0] >= min(p1[0], q1[0])
        and p2[1] <= max(p1[1], q1[1])
        and p2[1] >= min(p1[1], q1[1])
    ):
        return True

    if (
        o2 == 0
        and q2[0] <= max(p1[0], q1[0])
        and q2[0] >= min(p1[0], q1[0])
        and q2[1] <= max(p1[1], q1[1])
        and q2[1] >= min(p1[1], q1[1])
    ):
        return True

    if (
        o3 == 0
        and p1[0] <= max(p2[0], q2[0])
        and p1[0] >= min(p2[0], q2[0])
        and p1[1] <= max(p2[1], q2[1])
        and p1[1] >= min(p2[1], q2[1])
    ):
        return True

    if (
        o4 == 0
        and q1[0] <= max(p2[0], q2[0])
        and q1[0] >= min(p2[0], q2[0])
        and q1[1] <= max(p2[1], q2[1])
        and q1[1] >= min(p2[1], q2[1])
    ):
        return True

    return False


def line(point_1, point_2):
    """
    Create a line segment.

    Parameters
    ----------
    point_1 : tuple
        The first point of the line segment.
    point_2 : tuple
        The second point of the line segment.

    Returns
    -------
    tuple: The line segment.
    """
    return (point_1, point_2)


def find_segments(point, fairness_axis, acc_axis):
    """
    Find the segments of the polyline that the point projects onto.

    Parameters
    ----------
    point : tuple
        The point to be compared.
    fairness_axis : array, shape = (n_samples,)
        The array of the fairness axis.
    acc_axis : array, shape = (n_samples,)
        The array of the accuracy axis.

    Returns
    -------
    segment_x : tuple
        The segment of the polyline based on the x-coordinate.
    segment_y : tuple
        The segment of the polyline based on the y-coordinate.
    """
    polyline = list(zip(fairness_axis, acc_axis))
    # Sort the polyline points by x-coordinate
    polyline.sort(key=lambda p: p[0])

    # Initialize variables to store the segment points based on x and y coordinates
    segment_x = None, None
    segment_y = None, None

    # Find the polyline segment that the point projects onto
    for i in range(len(polyline) - 1):
        if polyline[i][0] <= point[0] <= polyline[i + 1][0]:
            # Store the segment points based on x-coordinate
            segment_x = polyline[i], polyline[i + 1]
            break

    # Sort the polyline points by y-coordinate for the y segment
    polyline.sort(key=lambda p: p[1])

    # Find the polyline segment that the point projects onto based on y-coordinate
    for i in range(len(polyline) - 1):
        if polyline[i][1] <= point[1] <= polyline[i + 1][1]:
            # Store the segment points based on y-coordinate
            segment_y = polyline[i], polyline[i + 1]
            break

    return segment_x, segment_y


def get_baseline_bounds(point, fairness_axis, acc_axis):
    """
    Get the segment bounds of the fairness and accuracy axis for the point with respect to the baseline.

    Parameters
    ----------
    point : tuple
        The point to be compared.
    fairness_axis : array, shape = (n_samples,)
        The array of the fairness axis.
    acc_axis : array, shape = (n_samples,)
        The array of the accuracy axis.

    Returns
    -------
    fair_segment : tuple
        The segment of the fairness axis.
    acc_segment : tuple
        The segment of the accuracy axis.
    """
    fair_point, acc_point = point
    pt_1 = get_closer_bounds(fair_point, fairness_axis, acc_axis, "lower")
    pt_2 = get_closer_bounds(fair_point, fairness_axis, acc_axis, "upper")
    fair_segment = line(pt_1, pt_2)
    pt_1 = get_closer_bounds(acc_point, acc_axis, fairness_axis, "lower")
    pt_2 = get_closer_bounds(acc_point, acc_axis, fairness_axis, "upper")
    acc_segment = line(pt_1, pt_2)
    return fair_segment, acc_segment


def get_indexes(x_segment, y_segment, x_axis, y_axis):
    """
    Get the indexes of the fairness and accuracy axis for the points between the x and y segments.

    Parameters
    ----------
    x_segment : tuple
        The segment of the fairness axis.
    y_segment : tuple
        The segment of the accuracy axis.
    x_axis : array, shape = (n_samples,)
        The array of the fairness axis.
    y_axis : array, shape = (n_samples,)
        The array of the accuracy axis.

    Returns
    -------
    indexes : array, shape = (n_samples,)
        The indexes of the fairness and accuracy axis.
    x_axis : array, shape = (n_samples,)
        The reversed array of the fairness axis.
    y_axis : array, shape = (n_samples,)
        The reversed array of the accuracy axis.
    """
    x_1 = max(x_segment[0][0], x_segment[1][0])
    x_2 = min(y_segment[0][0], y_segment[1][0])

    # reverse the order of the fairness axis
    x_axis = x_axis[::-1]
    y_axis = y_axis[::-1]

    # find the index of the x_1 and x_2 in the fairness_norm
    idx_1 = np.where(x_axis == x_1)[0][0]
    idx_2 = np.where(y_axis == x_2)[0][0]

    indexes = np.arange(idx_1, idx_2 + 1).astype(int)
    return indexes, x_axis, y_axis


def get_points(point, x_segment, y_segment):
    """
    Get the intersection points of the fairness and accuracy axis with the given point projection.

    Parameters
    ----------
    point : tuple
        The point to be compared.
    x_segment : array, shape = (n_samples,)
        The segment of the fairness axis.
    y_segment : array, shape = (n_samples,)
        The segment of the accuracy axis.

    Returns
    -------
    point_a : tuple
        The first intersection point.
    point_b : tuple
        The second intersection point.
    """
    a, b = get_rect_equation(x_segment[0], x_segment[1])
    c, d = get_rect_equation(y_segment[0], y_segment[1])

    point_a = (point[0], a * point[0] + b)
    point_b = ((point[1] - d) / c, point[1])
    return point_a, point_b


def get_area(point, x_axis, y_axis):
    """
    Get the area of the polygon formed by the baseline and the point.

    Parameters
    ----------
    point : tuple
        The point to be compared.
    x_axis : array, shape = (n_samples,)
        The array of the fairness axis.
    y_axis : array, shape = (n_samples,)
        The array of the accuracy axis.

    Returns
    -------
    total_area : float
        The area of the polygon.
    """
    fair_seg, acc_seg = get_baseline_bounds(point, x_axis, y_axis)
    if fair_seg == acc_seg:
        return (fair_seg[1][0] - point[0]) * (point[1] - fair_seg[0][1]) / 2
    point_a, point_b = get_points(point, fair_seg, acc_seg)
    indexes, x_axis, y_axis = get_indexes(fair_seg, acc_seg, x_axis, y_axis)
    p_initial = point_a
    total_area = 0
    for idx in indexes:
        p = (x_axis[idx], y_axis[idx])
        delta_x = p[0] - p_initial[0]
        delta_y = p[1] - p_initial[1]
        h = point[1] - p[1]
        triangle_area = 0.5 * delta_x * delta_y
        square_area = delta_x * h
        total_area += square_area + triangle_area
        p_initial = p

    delta_x = point_b[0] - p_initial[0]
    delta_y = point_b[1] - p_initial[1]
    triangle_area = 0.5 * delta_x * delta_y

    total_area += triangle_area
    return total_area
