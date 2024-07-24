def k_anonymity(df, qi):
    """

    Description
    -----------
    Computes k-Anonymity metric. K-anonymity is a property of a dataset that indicates the re-identifiability of its records. A dataset is
    k-anonymous if quasi-identifiers for each person in the dataset are identical to at least k - 1 other people also in the dataset.

    Parameters
    ----------

    df : pandas Dataframe
        input dataset

    qi : list or numpy array
        quasi identifiers

    Returns
    -------

    pd.Series: Computed metric.
    """
    return df[qi].value_counts()


def l_diversity(df, qi, sa):
    """

    Description
    -----------
    Computes l-Diversity metric. L-diversity is a property of a dataset and an extension of k-anonymity that measures the diversity of
    sensitive values for each column in which they occur. A dataset has l-diversity if, for every set of rows with identical
    quasi-identifiers, there are at least l distinct values for each sensitive attribute.

    Parameters
    ----------

    df : pandas Dataframe
        input dataset

    qi : list or numpy array
        quasi identifiers

    sa : list or numpy array
         sensitive attribute

    Returns
    -------

    dict[str, list]: Computed metric per sensitive attribute.
    """
    df_grouped = df.groupby(qi, as_index=False)
    return {s: sorted([len(row["unique"]) for _, row in df_grouped[s].agg(["unique"]).dropna().iterrows()]) for s in sa}
