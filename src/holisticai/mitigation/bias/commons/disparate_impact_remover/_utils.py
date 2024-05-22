import math
import random
from copy import deepcopy
from typing import Union

import numpy as np

from ._categorical_feature import CategoricalFeature
from ._sparse_list import SparseList


def get_categories_count_norm(
    categories, all_stratified_groups, count_dict, group_features
):
    """
    Description
    -----------
    Find the normalized count for each category, where normalized count is count divided by
    the number of people in that group.
    """
    norm = {
        cat: SparseList(
            data=(
                count_dict[cat][i] * (1.0 / len(group_features[group].data))
                if group_features[group].data
                else 0.0
                for i, group in enumerate(all_stratified_groups)
            )
        )
        for cat in categories
    }
    return norm


def gen_desired_dist(
    group_index, cat, col_id, median, repair_level, norm_counts, feature_to_remove, mode
):
    """
    Description
    -----------
    Generate the desired distribution and desired "count" for a given group-category-feature combination.
    """
    if feature_to_remove == col_id:
        return 1 if cat == mode else (1 - repair_level) * norm_counts[cat][group_index]
    else:
        return (1 - repair_level) * norm_counts[cat][group_index] + (
            repair_level * median[cat]
        )


def assign_overflow(
    all_stratified_groups, categories, overflow, group_features, repair_generator
):
    """
    Description
    -----------
    Assign overflow observations to categories based on the group's desired distribution.
    """
    feature = deepcopy(group_features)
    assigned_overflow = {}
    desired_dict_list = {}
    for group_index, group in enumerate(all_stratified_groups):
        # Calculate the category proportions.
        dist_generator = lambda cat: repair_generator(group_index, cat)
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
    Description
    -----------
    Count the observations in each category.
    """
    count_dict = {
        cat: SparseList(
            data=(
                group_feature[group].category_count[cat]
                if cat in group_feature[group].category_count
                else 0
                for group in all_stratified_groups
            )
        )
        for cat in categories
    }

    return count_dict


def gen_desired_count(
    group_index, group, category, median, group_features, repair_level, categories_count
):
    med = median[category]
    size = len(group_features[group].data)
    count = categories_count[category][group_index]
    des_count = math.floor(((1 - repair_level) * count) + (repair_level) * med * size)
    return des_count


def flow_on_group_features(all_stratified_groups, group_features, repair_generator):
    """
    Description
    -----------
    Run Max-flow to distribute as many observations to categories as possible.
    Overflow are those observations that are left over
    """
    dict1 = {}
    dict2 = {}
    for i, group in enumerate(all_stratified_groups):
        feature = group_features[group]
        count_generator = lambda category: repair_generator(i, group, category)
        DG = feature.create_graph(count_generator)
        new_feature, overflow = feature.repair(DG)
        dict2[group] = overflow
        dict1[group] = new_feature

    return dict1, dict2


def get_count(cat, group_feature_category_count):
    if cat in group_feature_category_count:
        return group_feature_category_count[cat]
    else:
        return 0


def get_count_norm(count, group_feature_data):
    if group_feature_data:
        return count * (1.0 / len(group_feature_data))
    else:
        return 0.0


def get_column_type(values: Union[list, np.ndarray]):
    if all(isinstance(value, float) for value in values):
        return float
    elif all(isinstance(value, int) for value in values):
        return int
    else:
        return str


def get_median(values, kdd):

    if not values:
        raise Exception("Cannot calculate median of list with no values!")

    sorted_values = deepcopy(values)
    sorted_values.sort()  # Not calling `sorted` b/c `sorted_values` may not be list.

    if kdd:
        return sorted_values[len(values) // 2]
    else:
        if len(values) % 2 == 0:
            return sorted_values[len(values) // 2 - 1]
        else:
            return sorted_values[len(values) // 2]


def get_group_data(all_stratified_groups, stratified_group_data, col_id):
    group_features = {}
    for group in all_stratified_groups:
        points = [
            (i, val)
            for indices, val in stratified_group_data[group][col_id]
            for i in indices
        ]
        points = sorted(points, key=lambda x: x[0])  # Sort by index
        values = [value for _, value in points]

        # send values to CategoricalFeature object, which bins the data into categories
        group_features[group] = CategoricalFeature(values)
    return group_features


# Find the median normalized count for each category
def get_median_per_category(categories, categories_count_norm):
    return {cat: get_median(categories_count_norm[cat], False) for cat in categories}


def get_mode(values):
    counts = {}
    for value in values:
        counts[value] = 1 if value not in counts else counts[value] + 1
    mode_tuple = max(list(counts.items()), key=lambda tup: tup[1])
    return mode_tuple[0]


def make_histogram_bins(bin_size_calculator, data, col_id):
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


def FreedmanDiaconisBinSize(feature_values):
    q75, q25 = np.percentile(feature_values, [75, 25])
    IQR = q75 - q25
    return 2.0 * IQR * len(feature_values) ** (-1.0 / 3.0)
