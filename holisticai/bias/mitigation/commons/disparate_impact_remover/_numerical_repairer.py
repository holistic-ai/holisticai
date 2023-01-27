from ._categorical_repairer import CategoricalRepairer
from ._utils import FreedmanDiaconisBinSize as bin_calculator
from ._utils import get_median, make_histogram_bins


class NumericalRepairer:
    def __init__(
        self, feature_to_repair, repair_level, kdd=False, features_to_ignore=[]
    ):
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level
        self.kdd = kdd
        self.features_to_ignore = features_to_ignore
        self.categoric_repairer = CategoricalRepairer(
            feature_to_repair=feature_to_repair,
            repair_level=repair_level,
            kdd=kdd,
            features_to_ignore=features_to_ignore,
        )

    def repair(self, data_to_repair):

        # Convert the "feature_to_repair" into a pseudo-categorical feature by
        # applying binning on that column.
        binned_data = [row[:] for row in data_to_repair]
        index_bins = make_histogram_bins(
            bin_calculator, data_to_repair, self.feature_to_repair
        )

        category_medians = {}
        for i, index_bin in enumerate(index_bins):
            bin_name = "BIN_{}".format(
                i
            )  # IE, the "category" to replace numeric values.
            for j in index_bin:
                binned_data[j][self.feature_to_repair] = bin_name
            category_vals = [
                data_to_repair[j][self.feature_to_repair] for j in index_bin
            ]
            category_medians[bin_name] = get_median(category_vals, self.kdd)

        repaired_data = self.categoric_repairer.repair(binned_data)

        # Replace the "feature_to_repair" column with the median numeric value.
        for i in range(len(repaired_data)):
            if self.repair_level > 0:
                rep_category = repaired_data[i][self.feature_to_repair]
                repaired_data[i][self.feature_to_repair] = category_medians[
                    rep_category
                ]
            else:
                repaired_data[i][self.feature_to_repair] = data_to_repair[i][
                    self.feature_to_repair
                ]
        return repaired_data
