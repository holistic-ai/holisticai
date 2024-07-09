from collections import defaultdict
from itertools import product

from holisticai.bias.mitigation.commons.disparate_impact_remover._categorical_feature import CategoricalFeature
from holisticai.bias.mitigation.commons.disparate_impact_remover._utils import (
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
    def __init__(self, feature_to_repair, repair_level, kdd=False, features_to_ignore=None):
        """
        Initializes a CategoricalRepairer object.

        Parameters
        ----------

        feature_to_repair: str
            The name of the feature to repair.
        repair_level: float
            The desired repair level, ranging from 0 to 1.
        kdd: (bool, optional)
            Whether to use the KDD repair method. Defaults to False.
        features_to_ignore: (list, optional)
            A list of features to ignore during repair. Defaults to None.
        """
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level
        self.kdd = kdd
        self.features_to_ignore = [] if features_to_ignore is None else features_to_ignore
        self.preprocessor = DataPreprocessor(self.feature_to_repair, self.kdd, self.features_to_ignore)

    def repair(self, data_to_repair):
        """
        Repairs the given data by applying bias mitigation techniques.

        Parameters
        ----------
        data_to_repair: list
            The data to be repaired.

        Returns:
            list: The repaired data.
        """
        (
            col_ids,
            cols_to_repair,
            data_dict,
            repair_types,
            unique_col_vals,
            index_lookup,
            all_stratified_groups,
            stratified_group_indices,
            val_sets,
            stratified_group_data,
        ) = self.preprocessor.preprocessing(data_to_repair)

        mode_feature_to_repair = get_mode(data_dict[self.feature_to_repair])

        for col_id in cols_to_repair:
            col = data_dict[col_id]
            repair_type = repair_types[col_id]

            if repair_type in {int, float}:
                NumericRepairer(self.repair_level, self.kdd).repair(
                    col,
                    data_dict,
                    col_id,
                    unique_col_vals,
                    index_lookup,
                    all_stratified_groups,
                    stratified_group_data,
                    val_sets,
                )

            elif repair_type in {str}:
                StringRepairer(self.feature_to_repair, self.repair_level).repair(
                    col,
                    data_dict,
                    col_id,
                    all_stratified_groups,
                    stratified_group_indices,
                    stratified_group_data,
                    mode_feature_to_repair,
                )

        repaired_data = []
        for i, orig_row in enumerate(data_to_repair):
            new_row = [orig_row[j] if j not in cols_to_repair else data_dict[j][i] for j in col_ids]
            repaired_data.append(new_row)

        return repaired_data


class DataPreprocessor:
    def __init__(self, feature_to_repair, kdd, features_to_ignore):
        """
        Initializes a CategoricalRepairer object.

        Parameters
        ----------
        feature_to_repair: str
            The name of the feature to repair.
        kdd: float
            The KDD value used for repairing the feature.
        features_to_ignore: list
            A list of features to ignore during the repair process.
        """
        self.feature_to_repair = feature_to_repair
        self.kdd = kdd
        self.features_to_ignore = features_to_ignore

    def preprocessing(self, data_to_repair):
        """
        Preprocesses the data for repair.

        Parameters
        ----------

        data_to_repair: list
            The data to be repaired.
        """
        num_cols = len(data_to_repair[0])
        col_ids = list(range(num_cols))

        col_types = [
            "Y" if i not in self.features_to_ignore else "I" if i != self.feature_to_repair else "X" for i in col_ids
        ]
        col_type_dict = dict(zip(col_ids, col_types))
        not_I_col_ids = [x for x in col_ids if col_type_dict[x] != "I"]

        cols_to_repair = (
            [x for x in col_ids if col_type_dict[x] in "YX"]
            if not self.kdd
            else [x for x in col_ids if col_type_dict[x] == "Y"]
        )

        safe_stratify_cols = [self.feature_to_repair]

        data_dict = {col_id: [] for col_id in col_ids}
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
            col_values = sorted(set(col_values))
            unique_col_vals[col_id] = col_values
            col_values_len = len(col_values)
            index_lookup[col_id] = {col_values[i]: i for i in range(col_values_len)}

        unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]

        stratified_group_indices = defaultdict(list)
        val_sets = defaultdict(lambda: defaultdict(set))
        for i, row in enumerate(data_to_repair):
            group = tuple(row[col] for col in safe_stratify_cols)
            for col_id in cols_to_repair:
                val_sets[group][col_id].add(row[col_id])
            stratified_group_indices[group].append(i)

        all_stratified_groups = list(product(*unique_stratify_values))

        stratified_group_data = StratifiedGroupDataComputer().compute(
            data_dict, stratified_group_indices, all_stratified_groups
        )

        return (
            col_ids,
            cols_to_repair,
            data_dict,
            repair_types,
            unique_col_vals,
            index_lookup,
            all_stratified_groups,
            stratified_group_indices,
            val_sets,
            stratified_group_data,
        )


class StratifiedGroupDataComputer:
    def compute(self, data_dict, stratified_group_indices, all_stratified_groups):
        """
        Computes the stratified group data based on the given data dictionary, stratified group indices, and all stratified groups.

        Paramters
        ---------

        data_dict: dict
            A dictionary containing the data for each column.
        stratified_group_indices: dict
            A dictionary containing the indices of each stratified group.
        all_stratified_groups: list
            A list of all stratified groups.

        Returns:
            dict: A dictionary containing the stratified group data for each group and column.
        """
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


class StringRepairer:
    def __init__(self, feature_to_repair, repair_level):
        """
        Initialize the CategoricalRepairer object.

        Parameters
        ---------

        feature_to_repair: str
            The name of the feature to repair.
        repair_level: float
            The repair level to be applied to the feature.
        """
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level

    def repair(
        self,
        col,
        data_dict,
        col_id,
        all_stratified_groups,
        stratified_group_indices,
        stratified_group_data,
        mode_feature_to_repair,
    ):
        """
        Repairs the categorical feature values in the given column based on the provided data.

        Parameters
        ----------

        col: str
            The name of the column to repair.
        data_dict: dict
            A dictionary containing the data.
        col_id: int
            The ID of the column to repair.
        all_stratified_groups: list
            A list of all stratified groups.
        stratified_group_indices: dict
            A dictionary mapping each stratified group to its corresponding indices.
        stratified_group_data: dict
            A dictionary containing the data for each stratified group.
        mode_feature_to_repair: str
            The mode feature to repair.
        """
        feature = CategoricalFeature(col)
        categories = list(feature.bin_index_dict.keys())

        group_features = get_group_data(all_stratified_groups, stratified_group_data, col_id)
        categories_count = get_categories_count(categories, all_stratified_groups, group_features)
        categories_count_norm = get_categories_count_norm(
            categories, all_stratified_groups, categories_count, group_features
        )
        median = get_median_per_category(categories, categories_count_norm)

        def dist_generator(group_index, category):
            return gen_desired_dist(
                group_index,
                category,
                col_id,
                median,
                self.repair_level,
                categories_count_norm,
                self.feature_to_repair,
                mode_feature_to_repair,
            )

        def count_generator(group_index, group, category):
            return gen_desired_count(
                group_index, group, category, median, group_features, self.repair_level, categories_count
            )

        group_features, overflow = flow_on_group_features(all_stratified_groups, group_features, count_generator)
        group_features, _, _ = assign_overflow(
            all_stratified_groups, categories, overflow, group_features, dist_generator
        )

        repaired_values = {group: group_features[group].data for group in all_stratified_groups}

        for group in all_stratified_groups:
            indices = stratified_group_indices[group]
            group_repaired_values = repaired_values[group]
            for i, index in enumerate(indices):
                data_dict[col_id][index] = group_repaired_values[i]


class NumericRepairer:
    def __init__(self, repair_level, kdd):
        """
        Initializes a new instance of the CategoricalRepairer class.

        Parameters
        ----------

        repair_level: float
            The repair level to be applied.
        kdd: bool
            A flag indicating whether to use KDD repair or not.
        """
        self.repair_level = repair_level
        self.kdd = kdd

    def repair(
        self,
        col,
        data_dict,
        col_id,
        unique_col_vals,
        index_lookup,
        all_stratified_groups,
        stratified_group_data,
        val_sets,
    ):
        """
        Repairs the categorical column values based on the provided parameters.

        Parameters
        ----------

        col: list
            The list of column values.
        data_dict: dict
            The dictionary containing the data.
        col_id: int
            The ID of the column to be repaired.
        unique_col_vals: dict
            The dictionary containing unique column values.
        index_lookup: dict
            The dictionary containing the index lookup.
        all_stratified_groups: list
            The list of all stratified groups.
        stratified_group_data: dict
            The dictionary containing stratified group data.
        val_sets: dict
            The dictionary containing validation sets.

        Returns:
            None
        """
        num_quantiles = min(len(val_sets[group][col_id]) for group in all_stratified_groups)
        quantile_unit = 1.0 / num_quantiles
        group_offsets = {group: 0 for group in all_stratified_groups}
        quantiles_data = []

        for _ in range(num_quantiles):
            median_at_quantiles = []
            indices_per_group = {}

            for group in all_stratified_groups:
                group_data_at_col = stratified_group_data[group][col_id]
                num_vals = len(group_data_at_col)
                group_offset = group_offsets[group] * num_vals
                offset = int(round(group_offset))
                number_to_get = int(round(group_offset + quantile_unit * num_vals)) - offset
                group_offsets[group] += quantile_unit

                if number_to_get > 0:
                    offset_data = group_data_at_col[offset : offset + number_to_get]
                    indices_per_group[group] = [i for val_indices, _ in offset_data for i in val_indices]
                    values = sorted(float(val) for _, val in offset_data)
                    median_at_quantiles.append(get_median(values, self.kdd))

            median = get_median(median_at_quantiles, self.kdd)
            quantiles_data.append((median, indices_per_group))

        median_val_pos = {q[0]: index_lookup[col_id][q[0]] for q in quantiles_data}

        for median, indices_per_group in quantiles_data:
            median_pos = median_val_pos[median]
            for group in all_stratified_groups:
                indices_to_repair = indices_per_group.get(group, [])
                if not indices_to_repair:
                    continue

                original_values = [col[index] for index in indices_to_repair]
                current_val_positions = [index_lookup[col_id][value] for value in original_values]
                distances_to_repair = [
                    int(round((median_pos - position) * self.repair_level)) for position in current_val_positions
                ]
                indices_of_repair_values = [
                    current_val_positions[i] + distances_to_repair[i] for i in range(len(indices_to_repair))
                ]
                repaired_values = [unique_col_vals[col_id][index] for index in indices_of_repair_values]

                for i in range(len(indices_to_repair)):
                    data_dict[col_id][indices_to_repair[i]] = repaired_values[i]
