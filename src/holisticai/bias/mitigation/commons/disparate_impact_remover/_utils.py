from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Union

import numpy as np
from holisticai.bias.mitigation.commons.disparate_impact_remover._categorical_feature import CategoricalFeature
from holisticai.bias.mitigation.commons.disparate_impact_remover._sparse_list import SparseList


def get_categories_count_norm(categories, all_stratified_groups, count_dict, group_features):
    """
    Get the normalized count for each category , where normalized count is count divided by
    the number of people in that group.


    Parameters
    ----------
    categories : list
        The list of categories.
    all_stratified_groups : list
        The list of stratified groups.
    count_dict : dict
        The dictionary containing the count of observations in each category.
    group_features : dict
        The dictionary containing the CategoricalFeature objects for each group.

    Returns
    -------
    dict
        The dictionary containing the normalized count for each category.
    """
    norm = {
        cat: SparseList(
            data=(
                count_dict[cat][i] * (1.0 / len(group_features[group].data)) if group_features[group].data else 0.0
                for i, group in enumerate(all_stratified_groups)
            )
        )
        for cat in categories
    }
    return norm


def gen_desired_dist(group_index, cat, col_id, median, repair_level, norm_counts, feature_to_remove, mode):
    """
    Generate the desired distribution for each category in a group.

    Parameters
    ----------
    group_index : int
        The index of the group.
    cat : str
        The category.
    col_id : int
        The column index.
    median : dict
        The dictionary containing the median value for each category.
    repair_level : float
        The repair level.
    norm_counts : dict
        The dictionary containing the normalized count for each category.
    feature_to_remove : int
        The feature to remove.
    mode : str
        The mode value.

    Returns
    -------
    float
        The desired distribution for the category.
    """
    if feature_to_remove == col_id:
        return 1 if cat == mode else (1 - repair_level) * norm_counts[cat][group_index]
    return (1 - repair_level) * norm_counts[cat][group_index] + (repair_level * median[cat])


def assign_overflow(all_stratified_groups, categories, overflow, group_features, repair_generator):
    """
    Assign the overflow to the desired categories based on the group's desired distribution.

    Parameters
    ----------
    all_stratified_groups : list
        The list of stratified groups.
    categories : list
        The list of categories.
    overflow : dict
        The dictionary containing the overflow for each group.
    group_features : dict
        The dictionary containing the CategoricalFeature objects for each group.
    repair_generator : function
        The repair generator function.

    Returns
    -------
    dict
        The dictionary containing the assigned overflow for each group.
    dict
        The dictionary containing the desired distribution for each category.
    dict
        The dictionary containing the feature after overflow assignment.
    """
    feature = deepcopy(group_features)
    assigned_overflow = {}
    desired_dict_list = {}
    for group_index, group in enumerate(all_stratified_groups):
        # Calculate the category proportions.
        def dist_generator(cat):
            return repair_generator(group_index, cat)

        cat_props = list(map(dist_generator, categories))

        if all(elem == 0 for elem in cat_props):  # TODO: Check that this is correct!
            cat_props = [1.0 / len(cat_props)] * len(cat_props)
        s = float(sum(cat_props))
        cat_props = [elem / s for elem in cat_props]
        desired_dict_list[group] = cat_props
        assigned_overflow[group] = {}
        for i in range(int(overflow[group])):
            distribution_list = desired_dict_list[group]
            number = random.uniform(0, 1)
            cat_index = 0
            tally = 0
            for j in range(len(distribution_list)):
                value = distribution_list[j]
                if number < (tally + value):
                    cat_index = j
                    break
                tally += value
            assigned_overflow[group][i] = categories[cat_index]
        # Actually do the assignment
        count = 0
        for i, value in enumerate(group_features[group].data):
            if value == 0:
                (feature[group].data)[i] = assigned_overflow[group][count]
                count += 1
    return feature, assigned_overflow, desired_dict_list


def get_categories_count(categories, all_stratified_groups, group_feature):
    """
    Get the count for each category.

    Parameters
    ----------
    categories : list
        The list of categories.
    all_stratified_groups : list
        The list of stratified groups.
    group_feature : dict
        The dictionary containing the CategoricalFeature objects for each group.

    Returns
    -------
    dict
        The dictionary containing the count for each category.
    """
    count_dict = {
        cat: SparseList(
            data=(
                group_feature[group].category_count[cat] if cat in group_feature[group].category_count else 0
                for group in all_stratified_groups
            )
        )
        for cat in categories
    }

    return count_dict


def gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count):
    """
    Generate the desired count for each category in a group.

    Parameters
    ----------
    group_index : int
        The index of the group.
    group : str
        The group.
    category : str
        The category.
    median : dict
        The dictionary containing the median value for each category.
    group_features : dict
        The dictionary containing the CategoricalFeature objects for each group.
    repair_level : float
        The repair level.
    categories_count : dict
        The dictionary containing the count for each category.

    Returns
    -------
    int
        The desired count for the category.
    """
    med = median[category]
    size = len(group_features[group].data)
    count = categories_count[category][group_index]
    des_count = math.floor(((1 - repair_level) * count) + (repair_level) * med * size)
    return des_count


def flow_on_group_features(all_stratified_groups, group_features, repair_generator):
    """
    Flow on group features. Run Max-flow to distribute as many observations to categories as possible.
    Overflow are those observations that are left over after max-flow.

    Parameters
    ----------
    all_stratified_groups : list
        The list of stratified groups.
    group_features : dict
        The dictionary containing the CategoricalFeature objects for each group.
    repair_generator : function
        The repair generator function.

    Returns
    -------
    dict
        The dictionary containing the new feature after repair.
    dict
        The dictionary containing the overflow for each group.Description
    """
    dict1 = {}
    dict2 = {}
    for i, group in enumerate(all_stratified_groups):
        feature = group_features[group]

        def count_generator(category):
            return repair_generator(i, group, category)

        DG = feature.create_graph(count_generator)
        new_feature, overflow = feature.repair(DG)
        dict2[group] = overflow
        dict1[group] = new_feature

    return dict1, dict2


def get_count(cat, group_feature_category_count):
    """
    Get the count for a category.

    Parameters
    ----------
    cat : str
        The category.
    group_feature_category_count : dict
        The dictionary containing the count for each category.

    Returns
    -------
    int
        The count for the category.
    """
    if cat in group_feature_category_count:
        return group_feature_category_count[cat]
    return 0


def get_count_norm(count, group_feature_data):
    """
    Get the normalized count for a category.

    Parameters
    ----------
    count : int
        The count for the category.
    group_feature_data : list
        The list of data for the category.

    Returns
    -------
    float
        The normalized count for the category.
    """
    if group_feature_data:
        return count * (1.0 / len(group_feature_data))
    return 0.0


def get_column_type(values: Union[list, np.ndarray]):
    """
    Get the type of the column.

    Parameters
    ----------
    values : list or np.ndarray
        The list of values.

    Returns
    -------
    type
        The type of the column.
    """
    if all(isinstance(value, float) for value in values):
        return float

    if all(isinstance(value, int) for value in values):
        return int
    return str


def get_median(values, kdd):
    """
    Get the median of a list of values.

    Parameters
    ----------
    values : list
        The list of values.
    kdd : bool
        Whether to use the KDD method.

    Returns
    -------
    float
        The median of the list of values.
    """
    if not values:
        raise ValueError("Cannot calculate median of list with no values!")

    sorted_values = deepcopy(values)
    sorted_values.sort()  # Not calling `sorted` b/c `sorted_values` may not be list.

    if kdd:
        return sorted_values[len(values) // 2]

    if len(values) % 2 == 0:
        return sorted_values[len(values) // 2 - 1]

    return sorted_values[len(values) // 2]


def get_group_data(all_stratified_groups, stratified_group_data, col_id):
    """
    Get the group data for a column.

    Parameters
    ----------
    all_stratified_groups : list
        The list of stratified groups.
    stratified_group_data : dict
        The dictionary containing the stratified group data.
    col_id : int
        The column index.

    Returns
    -------
    dict
        The dictionary containing the group data for the column.
    """
    group_features = {}
    for group in all_stratified_groups:
        points = [(i, val) for indices, val in stratified_group_data[group][col_id] for i in indices]
        points = sorted(points, key=lambda x: x[0])  # Sort by index
        values = [value for _, value in points]

        # send values to CategoricalFeature object, which bins the data into categories
        group_features[group] = CategoricalFeature(values)
    return group_features


# Find the median normalized count for each category
def get_median_per_category(categories, categories_count_norm):
    """
    Get the median normalized count for each category.

    Parameters
    ----------
    categories : list
        The list of categories.
    categories_count_norm : dict
        The dictionary containing the normalized count for each category.

    Returns
    -------
    dict
        The dictionary containing the median normalized count for each category.
    """
    return {cat: get_median(categories_count_norm[cat], False) for cat in categories}


def get_mode(values):
    """
    Get the mode of a list of values.

    Parameters
    ----------
    values : list
        The list of values.

    Returns
    -------
    any
        The mode of the list of values.
    """
    counts = {}
    for value in values:
        counts[value] = 1 if value not in counts else counts[value] + 1
    mode_tuple = max(list(counts.items()), key=lambda tup: tup[1])
    return mode_tuple[0]


def make_histogram_bins(bin_size_calculator, data, col_id):
    """
    Make histogram bins for a column.

    Parameters
    ----------
    bin_size_calculator : function
        The bin size calculator function.
    data : list
        The input dataset.
    col_id : int
        The column index.

    Returns
    -------
    list
        The list of histogram bins.
    """
    feature_vals = [row[col_id] for row in data]
    bin_range = bin_size_calculator(feature_vals)

    if bin_range == 0.0:
        bin_range = 1.0

    data_tuples = list(enumerate(data))  # [(0,row), (1,row'), (2,row''), ... ]
    sorted_data_tuples = sorted(data_tuples, key=lambda tup: tup[1][col_id])

    max_val = max(data, key=lambda datum: datum[col_id])[col_id]
    min_val = min(data, key=lambda datum: datum[col_id])[col_id]

    index_bins = []
    val_ranges = []
    curr = min_val
    while curr <= max_val:
        index_bins.append([])
        val_ranges.append((curr, curr + bin_range))
        curr += bin_range

    for row_index, row in sorted_data_tuples:
        for bin_num, val_range in enumerate(val_ranges):
            if val_range[0] <= row[col_id] < val_range[1]:
                index_bins[bin_num].append(row_index)
                break

    index_bins = [b for b in index_bins if b]

    return index_bins


def freedman_diaconis_bin_size(feature_values):
    """
    Calculate the bin size using the Freedman-Diaconis rule.

    Parameters
    ----------
    feature_values : list
        The list of feature values.

    Returns
    -------
    float
        The bin size.
    """
    q75, q25 = np.percentile(feature_values, [75, 25])
    IQR = q75 - q25
    return 2.0 * IQR * len(feature_values) ** (-1.0 / 3.0)
