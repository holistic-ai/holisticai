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

import numpy as np
from collections import defaultdict

class CategoricalRepairer:
    def __init__(
        self, feature_to_repair, repair_level, kdd=False, features_to_ignore=[]
    ):
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level
        self.kdd = kdd
        self.features_to_ignore = features_to_ignore

    def repair(self, data_to_repair):
        col_ids, cols_to_repair, data_dict, repair_types, unique_col_vals, index_lookup, all_stratified_groups, stratified_group_indices, val_sets, stratified_group_data = self.preprocessing(data_to_repair)
        
        mode_feature_to_repair = get_mode(data_dict[self.feature_to_repair])
        
        for col_id in cols_to_repair:
            col = data_dict[col_id]
            repair_type = repair_types[col_id]
            
            if repair_type in {int, float}:
                self.numeric_repair(col, data_dict, col_id, unique_col_vals, index_lookup, all_stratified_groups, stratified_group_data, val_sets)
                
            elif repair_type in {str}:
                self.string_repair(col, data_dict, col_id, all_stratified_groups, stratified_group_indices, stratified_group_data, mode_feature_to_repair)
        
        repaired_data = []
        for i, orig_row in enumerate(data_to_repair):
            new_row = [orig_row[j] if j not in cols_to_repair else data_dict[j][i] for j in col_ids]
            repaired_data.append(new_row)
            
        return repaired_data
    
    def preprocessing(self, data_to_repair):
        num_cols = len(data_to_repair[0])
        col_ids = list(range(num_cols))

        # Get column type information
        col_types = ["Y" if i not in self.features_to_ignore else "I" if i != self.feature_to_repair else "X" for i in col_ids]

        col_type_dict = {col_id: col_type for col_id, col_type in zip(col_ids, col_types)}

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
            col_values = sorted(set(col_values))
            unique_col_vals[col_id] = col_values
            # look up a value, get its position
            col_values_len = len(col_values)
            index_lookup[col_id] = {col_values[i]: i for i in range(col_values_len)}

        unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
        
                
        # look up a stratified group, and get a list of indices corresponding to that group in the data
        stratified_group_indices = defaultdict(list)
        val_sets = defaultdict(lambda: defaultdict(set))
        for i, row in enumerate(data_to_repair):
            group = tuple(row[col] for col in safe_stratify_cols)
            for col_id in cols_to_repair:
                val_sets[group][col_id].add(row[col_id])
            stratified_group_indices[group].append(i)
        
        all_stratified_groups = list(product(*unique_stratify_values))
        
        stratified_group_data = self.compute_stratified_group_data(data_dict, stratified_group_indices, all_stratified_groups)
        
        return col_ids,cols_to_repair,data_dict,repair_types,unique_col_vals,index_lookup,all_stratified_groups,stratified_group_indices,val_sets,stratified_group_data

    def compute_stratified_group_data(self, data_dict, stratified_group_indices, all_stratified_groups):
        stratified_group_data = defaultdict(dict)
        for group in all_stratified_groups:
            for col_id, col_dict in data_dict.items():
                indices = defaultdict(list)
                for i in stratified_group_indices[group]:
                    value = col_dict[i]
                    indices[value].append(i)
                stratified_col_values = sorted((occurs, val) for val, occurs in indices.items())
                stratified_col_values.sort(key=lambda tup: tup[1])
                stratified_group_data[group][col_id] = stratified_col_values
        return stratified_group_data

    def string_repair(self, col, data_dict, col_id, all_stratified_groups, stratified_group_indices, stratified_group_data, mode_feature_to_repair):
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

    def numeric_repair(self, col, data_dict, col_id, unique_col_vals, index_lookup, all_stratified_groups, stratified_group_data, val_sets):
                        
        num_quantiles = min(
                len(val_sets[group][col_id]) for group in all_stratified_groups
            )
        quantile_unit = 1.0 / num_quantiles
        group_offsets = {group: 0 for group in all_stratified_groups}
        for quantile in range(num_quantiles):
            median_at_quantiles = []
            indices_per_group = {}
                                        
            for group in all_stratified_groups:
                group_data_at_col = stratified_group_data[group][col_id]
                num_vals = len(group_data_at_col)
                group_offset_times_num_vals = group_offsets[group] * num_vals
                offset = int(round(group_offset_times_num_vals))
                number_to_get = int(round(group_offset_times_num_vals + quantile_unit * num_vals)) - offset
                group_offsets[group] += quantile_unit

                if number_to_get > 0:
                    offset_data = group_data_at_col[offset : offset + number_to_get]
                    indices_per_group[group] = [i for val_indices, _ in offset_data for i in val_indices]
                    values = sorted(float(val) for _, val in offset_data)

                    # Find this group's median value at this quantile
                    median_at_quantiles.append(get_median(values, self.kdd))

            # Find the median value of all groups at this quantile (chosen from each group's medians)
            median = get_median(median_at_quantiles, self.kdd)
            median_val_pos = index_lookup[col_id][median]

            # Update values to repair the dataset.
            for group in all_stratified_groups:
                indices_to_repair = [index for index in indices_per_group[group]]
                original_values = [col[index] for index in indices_to_repair]
                current_val_positions = [index_lookup[col_id][value] for value in original_values]
                distances = [median_val_pos - position for position in current_val_positions]
                distances_to_repair = [int(round(distance * self.repair_level)) for distance in distances]
                indices_of_repair_values = [current_val_positions[i] + distances_to_repair[i] for i in range(len(indices_to_repair))]
                repaired_values = [unique_col_vals[col_id][index] for index in indices_of_repair_values]
                for i in range(len(indices_to_repair)):
                    data_dict[col_id][indices_to_repair[i]] = repaired_values[i]
   