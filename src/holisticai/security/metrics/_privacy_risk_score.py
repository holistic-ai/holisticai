import numpy as np


def _log_value(probs, small_value=1e-30):
    """
    Compute the negative logarithm of the given probabilities, ensuring numerical stability.

    Parameters
    ----------
    probs : array-like
        The probabilities for which to compute the negative logarithm.
    small_value : float, optional
        A small value to ensure numerical stability by preventing log(0). Default is 1e-30.

    Returns
    -------
    array-like
        The negative logarithm of the input probabilities, with numerical stability adjustments.
    """
    return -np.log(np.maximum(probs, small_value))


def _modified_entropy(probs, true_labels):
    """
    Compute the modified entropy for a set of probabilities and true labels.

    The modified entropy is calculated by adjusting the probabilities and their
    logarithms for the true labels, then computing the entropy using these modified
    values.

    Parameters
    ----------
    probs : ndarray
        Array of predicted probabilities with shape (n_samples, n_classes).
    true_labels : ndarray
        Array of true labels with shape (n_samples,).

    Returns
    -------
    ndarray
        Array of modified entropy values for each sample with shape (n_samples,).
    """
    log_probs = _log_value(probs)
    reverse_probs = 1 - probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)


def distrs_compute(tr_values, te_values, tr_labels, te_labels, num_bins=5, log_bins=True):
    """
    Compute the distributions of training and test values for each class.

    Parameters
    ----------
    tr_values : array-like
        Array of training values.
    te_values : array-like
        Array of test values.
    tr_labels : array-like
        Array of labels corresponding to the training values.
    te_labels : array-like
        Array of labels corresponding to the test values.
    num_bins : int, optional
        Number of bins to use for the histograms (default is 5).
    log_bins : bool, optional
        If True, use logarithmic bins; otherwise, use linear bins (default is True).

    Returns
    -------
    tr_distrs : ndarray
        Array of normalized histograms for the training values, one per class.
    te_distrs : ndarray
        Array of normalized histograms for the test values, one per class.
    all_bins : ndarray
        Array of bin edges used for the histograms, one set per class.
    """

    num_classes = len(set(tr_labels))
    tr_distrs, te_distrs, all_bins = [], [], []

    for i in range(num_classes):
        tr_list, te_list = tr_values[tr_labels == i], te_values[te_labels == i]
        if log_bins:
            # when using log scale, avoid very small number close to 0
            small_delta = 1e-10
            tr_list[tr_list <= small_delta] = small_delta
            te_list[te_list <= small_delta] = small_delta
        all_list = np.concatenate((tr_list, te_list))
        max_v, min_v = np.amax(all_list), np.amin(all_list)

        if log_bins:
            bins = np.logspace(np.log10(min_v), np.log10(max_v), num_bins + 1)
        else:
            bins = np.linspace(min_v, max_v, num_bins + 1)

        h1, _ = np.histogram(tr_list, bins=bins)
        h1 = h1 / float(len(tr_list))  # Normalize the histogram

        h2, _ = np.histogram(te_list, bins=bins)
        h2 = h2 / float(len(te_list))  # Normalize the histogram
        tr_distrs.append(h1)
        te_distrs.append(h2)
        all_bins.append(bins)

    tr_distrs, te_distrs, all_bins = np.array(tr_distrs), np.array(te_distrs), np.array(all_bins)
    return tr_distrs, te_distrs, all_bins


def find_index(bins, value):
    """
    Determine the bin index for a given value.

    For a given list of bin edges and a value, this function returns the index
    of the bin that includes the value. If the value is larger than the largest
    bin edge, it assigns the last bin. If the value is smaller than the smallest
    bin edge, it assigns the first bin.

    Parameters
    ----------
    bins : list or array-like
        A list or array of bin edges. The length of this list should be n+1 for n bins.
    value : float
        The value to be assigned to a bin.

    Returns
    -------
    int
        The index of the bin that includes the value.

    Examples
    --------
    >>> bins = [0, 1, 2, 3, 4]
    >>> find_index(bins, 2.5)
    2
    >>> find_index(bins, -1)
    0
    >>> find_index(bins, 5)
    3
    """
    # for given n bins (n+1 list) and one value, return which bin includes the value
    if value >= bins[-1]:
        return len(bins) - 2  # when value is larger than any bins, we assign the last bin
    if value <= bins[0]:
        return 0  # when value is smaller than any bins, we assign the first bin
    return np.argwhere(bins <= value)[-1][0]


def score_calculate(tr_distr, te_distr, ind):
    """
    Calculate the score based on training and testing distributions.

    The function computes the score as the ratio of the training distribution
    value to the sum of training and testing distribution values at a given index.
    If both distributions have zero probabilities at the given index, it searches
    for the nearest bin with a non-zero probability.

    Parameters
    ----------
    tr_distr : list or numpy.ndarray
        The training distribution values.
    te_distr : list or numpy.ndarray
        The testing distribution values.
    ind : int
        The index at which to calculate the score.

    Returns
    -------
    float
        The calculated score. If both distributions have zero probabilities at the
        given index, the score is calculated from the nearest bin with non-zero
        probability.
    """
    if tr_distr[ind] + te_distr[ind] != 0:
        return tr_distr[ind] / (tr_distr[ind] + te_distr[ind])
    # when both distributions have 0 probabilities, we find the nearest bin with non-zero probability
    for t_n in range(1, len(tr_distr)):
        t_ind = ind - t_n
        if t_ind >= 0 and tr_distr[t_ind] + te_distr[t_ind] != 0:
            return tr_distr[t_ind] / (tr_distr[t_ind] + te_distr[t_ind])
        t_ind = ind + t_n
        if t_ind < len(tr_distr) and tr_distr[t_ind] + te_distr[t_ind] != 0:
            return tr_distr[t_ind] / (tr_distr[t_ind] + te_distr[t_ind])
    return None


def _risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels):
    """
    Compute the privacy risk score for training points of the target classifier.
    Given training and test distributions (obtained from the shadow classifier),
    this function computes the corresponding privacy risk score for training points
    of the target classifier.

    Parameters
    ----------
    tr_distrs : list of np.ndarray
        List of training distributions for each class.
    te_distrs : list of np.ndarray
        List of test distributions for each class.
    all_bins : list of np.ndarray
        List of bin edges for each class.
    data_values : np.ndarray
        Array of data values for which the risk score is to be computed.
    data_labels : np.ndarray
        Array of labels corresponding to the data values.
    Returns
    -------
    np.ndarray
        Array of computed privacy risk scores for the given data values.
    """
    risk_score = []
    for i in range(len(data_values)):
        c_value, c_label = data_values[i], data_labels[i]
        c_tr_distr, c_te_distr, c_bins = tr_distrs[c_label], te_distrs[c_label], all_bins[c_label]
        c_index = find_index(c_bins, c_value)
        c_score = score_calculate(c_tr_distr, c_te_distr, c_index)
        risk_score.append(c_score)
    return np.array(risk_score)


def _check_input_format(input_data):
    """
    Check the input format for the probabilities and labels. Whether they are NumPy arrays or lists.
    If they are lists or array-like, convert them to NumPy arrays.

    Parameters
    ----------
    input_data : tuple
        Tuple containing the probabilities and labels.

    Returns
    -------
    tuple
        A tuple containing the converted probabilities and labels.
    """
    probabilities, labels = input_data
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    return probabilities, labels


def privacy_risk_score(shadow_train, shadow_test, target_train):
    """
    Calculate the risk score for membership inference attacks. The privacy risk score of an input sample\
    for the target machine learning model is deﬁned as the posterior probability that it is from the training set\
    after observing the target model's behavior over that sample. In other words, The privacy risk score estimates\
    an individual sample's probability of being in the target model's training set.

    A higher privacy risk score indicates a higher likelihood that the sample is in the target model's training set,\
    which implies a higher risk of membership inference attacks.

    Parameters
    ----------
    shadow_train : tuple
        A tuple containing the probabilities and labels for the shadow training set.
    shadow_test : tuple
        A tuple containing the probabilities and labels for the shadow test set.
    target_train : tuple
        A tuple containing the probabilities and labels for the target training set.

    Returns
    -------
    np.ndarray
        Array of computed privacy risk scores for the given data values.

    References
    ----------
    .. [1] Song, L., & Mittal, P. (2021). Systematic evaluation of privacy risks of machine learning models. In 30th USENIX Security Symposium (USENIX Security 21) (pp. 2615-2632).
    """
    shadow_train_probs, shadow_train_labels = _check_input_format(shadow_train)
    shadow_test_probs, shadow_test_labels = _check_input_format(shadow_test)
    target_train_probs, target_train_labels = _check_input_format(target_train)

    shadow_train_m_entr = _modified_entropy(shadow_train_probs, shadow_train_labels)
    shadow_test_m_entr = _modified_entropy(shadow_test_probs, shadow_test_labels)
    target_train_m_entr = _modified_entropy(target_train_probs, target_train_labels)

    tr_distrs, te_distrs, all_bins = distrs_compute(
        shadow_train_m_entr, shadow_test_m_entr, shadow_train_labels, shadow_test_labels, num_bins=5, log_bins=True
    )
    risk_score = _risk_score_compute(tr_distrs, te_distrs, all_bins, target_train_m_entr, target_train_labels)
    return risk_score
