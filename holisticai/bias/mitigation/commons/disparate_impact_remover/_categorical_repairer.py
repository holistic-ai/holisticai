from collections import defaultdict
from itertools import product

from ._categorical_feature import CategoricalFeature
from ._utils import (
    assign_overflow,
    flow_on_group_features,
    gen_desired_count,
    gen_desired_dist,
    get_categories_count,
    get_categories_count_norm,
    get_group_data,
    get_median,
    get_median_per_category,
    get_mode,
)


class CategoricalRepairer:
    def __init__(
        self, feature_to_repair, repair_level, kdd=False, features_to_ignore=[]
    ):
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level
        self.kdd = kdd
        self.features_to_ignore = features_to_ignore

    def repair(self, data_to_repair):
        num_cols = len(data_to_repair[0])
        col_ids = list(range(num_cols))

        # Get column type information
        col_types = ["Y"] * len(col_ids)
        for i, col in enumerate(col_ids):
            if i in self.features_to_ignore:
                col_types[i] = "I"
            elif i == self.feature_to_repair:
                col_types[i] = "X"

        col_type_dict = {
            col_id: col_type for col_id, col_type in zip(col_ids, col_types)
        }

        not_I_col_ids = [x for x in col_ids if col_type_dict[x] != "I"]

        if self.kdd:
            cols_to_repair = [x for x in col_ids if col_type_dict[x] == "Y"]
        else:
            cols_to_repair = [x for x in col_ids if col_type_dict[x] in "YX"]

        # To prevent potential perils with user-provided column names, map them to safe column names
        safe_stratify_cols = [self.feature_to_repair]

        # Extract column values for each attribute in data
        data_dict = {col_id: [] for col_id in col_ids}

        # Populate each attribute with its column values
        for row in data_to_repair:
            for i in col_ids:
                data_dict[i].append(row[i])

        repair_types = {}
        for col_id, values in list(data_dict.items()):
            if all(isinstance(value, float) for value in values):
                repair_types[col_id] = float
            elif all(isinstance(value, int) for value in values):
                repair_types[col_id] = int
            else:
                repair_types[col_id] = str

        unique_col_vals = {}
        index_lookup = {}
        for col_id in not_I_col_ids:
            col_values = data_dict[col_id]
            # extract unique values from column and sort
            col_values = sorted(list(set(col_values)))
            unique_col_vals[col_id] = col_values
            # look up a value, get its position
            index_lookup[col_id] = {col_values[i]: i for i in range(len(col_values))}

        unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
        all_stratified_groups = list(product(*unique_stratify_values))
        # look up a stratified group, and get a list of indices corresponding to that group in the data
        stratified_group_indices = defaultdict(list)

        # Find the number of unique values for each strat-group, organized per column.
        val_sets = {
            group: {col_id: set() for col_id in cols_to_repair}
            for group in all_stratified_groups
        }
        for i, row in enumerate(data_to_repair):
            group = tuple(row[col] for col in safe_stratify_cols)
            for col_id in cols_to_repair:
                val_sets[group][col_id].add(row[col_id])

            # Also remember that this row pertains to this strat-group.
            stratified_group_indices[group].append(i)

        stratified_group_data = {group: {} for group in all_stratified_groups}
        for group in all_stratified_groups:
            for col_id, col_dict in list(data_dict.items()):
                # Get the indices at which each value occurs.
                indices = {}
                for i in stratified_group_indices[group]:
                    value = col_dict[i]
                    if value not in indices:
                        indices[value] = []
                    indices[value].append(i)

                stratified_col_values = [
                    (occurs, val) for val, occurs in list(indices.items())
                ]
                stratified_col_values.sort(key=lambda tup: tup[1])
                stratified_group_data[group][col_id] = stratified_col_values

        mode_feature_to_repair = get_mode(data_dict[self.feature_to_repair])

        # Repair Data and retrieve the results
        for col_id in cols_to_repair:
            # which bucket value we're repairing
            group_offsets = {group: 0 for group in all_stratified_groups}
            col = data_dict[col_id]

            num_quantiles = min(
                len(val_sets[group][col_id]) for group in all_stratified_groups
            )
            quantile_unit = 1.0 / num_quantiles

            if repair_types[col_id] in {int, float}:
                for quantile in range(num_quantiles):
                    median_at_quantiles = []
                    indices_per_group = {}

                    for group in all_stratified_groups:
                        group_data_at_col = stratified_group_data[group][col_id]
                        num_vals = len(group_data_at_col)
                        offset = int(round(group_offsets[group] * num_vals))
                        number_to_get = int(
                            round((group_offsets[group] + quantile_unit) * num_vals)
                            - offset
                        )
                        group_offsets[group] += quantile_unit

                        if number_to_get > 0:

                            # Get data at this quantile from this Y column such that stratified X = group
                            offset_data = group_data_at_col[
                                offset : offset + number_to_get
                            ]
                            indices_per_group[group] = [
                                i for val_indices, _ in offset_data for i in val_indices
                            ]
                            values = sorted([float(val) for _, val in offset_data])

                            # Find this group's median value at this quantile
                            median_at_quantiles.append(get_median(values, self.kdd))

                    # Find the median value of all groups at this quantile (chosen from each group's medians)
                    median = get_median(median_at_quantiles, self.kdd)
                    median_val_pos = index_lookup[col_id][median]

                    # Update values to repair the dataset.
                    for group in all_stratified_groups:
                        for index in indices_per_group[group]:
                            original_value = col[index]

                            current_val_pos = index_lookup[col_id][original_value]
                            distance = (
                                median_val_pos - current_val_pos
                            )  # distance between indices
                            distance_to_repair = int(
                                round(distance * self.repair_level)
                            )
                            index_of_repair_value = current_val_pos + distance_to_repair
                            repaired_value = unique_col_vals[col_id][
                                index_of_repair_value
                            ]

                            # Update data to repaired valued
                            data_dict[col_id][index] = repaired_value

            # Categorical Repair is done below
            elif repair_types[col_id] in {str}:
                feature = CategoricalFeature(col)
                categories = list(feature.bin_index_dict.keys())

                group_features = get_group_data(
                    all_stratified_groups, stratified_group_data, col_id
                )

                categories_count = get_categories_count(
                    categories, all_stratified_groups, group_features
                )

                categories_count_norm = get_categories_count_norm(
                    categories, all_stratified_groups, categories_count, group_features
                )

                median = get_median_per_category(categories, categories_count_norm)

                # Partially fill-out the generator functions to simplify later calls.
                dist_generator = lambda group_index, category: gen_desired_dist(
                    group_index,
                    category,
                    col_id,
                    median,
                    self.repair_level,
                    categories_count_norm,
                    self.feature_to_repair,
                    mode_feature_to_repair,
                )

                count_generator = (
                    lambda group_index, group, category: gen_desired_count(
                        group_index,
                        group,
                        category,
                        median,
                        group_features,
                        self.repair_level,
                        categories_count,
                    )
                )

                group_features, overflow = flow_on_group_features(
                    all_stratified_groups, group_features, count_generator
                )

                group_features, assigned_overflow, distribution = assign_overflow(
                    all_stratified_groups,
                    categories,
                    overflow,
                    group_features,
                    dist_generator,
                )

                # Return our repaired feature in the form of our original dataset
                for group in all_stratified_groups:
                    indices = stratified_group_indices[group]
                    for i, index in enumerate(indices):
                        repaired_value = group_features[group].data[i]
                        data_dict[col_id][index] = repaired_value

        # Replace stratified groups with their mode value, to remove it's information
        repaired_data = []
        for i, orig_row in enumerate(data_to_repair):
            new_row = [
                orig_row[j] if j not in cols_to_repair else data_dict[j][i]
                for j in col_ids
            ]
            repaired_data.append(new_row)
        return repaired_data
