import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def get_indexes_from_names(df, names):
    """
    Description
    -----------
    Gets the indexes from the columns

    Parameters
    ----------
    df : pandas dataframe
        the complete dataframe

    Returns
    -------
    list : list of index
    """
    indexes = [df.columns.get_loc(item) for item in names]
    return indexes


def process_y(y):
    """
    Description
    -----------
    Converts from one-hot encoding to a list of labels

    Parameters
    ----------
    y : numpy array
        the labels matrix

    Returns
    -------
    y_processed (list of labels)
    """
    y_processed = [i for line in y for i, val in enumerate(line) if val == 1]
    return y_processed


def get_majority_class(y):
    """
    Description
    -----------
    Converts from one-hot encoding to a list of labels

    Parameters
    ----------
    y : list
        list of labels

    Returns
    -------
    The majority class and its accuracy
    """
    y = np.array(y).flatten()
    most_common_item = max(y, key=y.tolist().count)
    acc = np.count_nonzero(y == most_common_item) / len(y)
    return most_common_item, acc


def get_initial_solution(y):
    """
    Description
    -----------
    Calculates the accuracy of the majority class

    Parameters
    ----------
    y : numpy array
        the labels matrix

    Returns
    -------
    The majority class and its accuracy
    """
    y_processed = process_y(y)
    return get_majority_class(y_processed)


def remove_inconcsistency(x, y):
    """
    Description
    -----------
    Remove the inconsistencies

    Parameters
    ----------
    x : numpy array
        Dataset features
    y : numpy array
        Dataset labels

    Returns
    -------
    The dataset withtout the inconsistencies
    """
    x = x.tolist()
    y = y.tolist()

    new_x = []
    new_y = []

    for i in range(len(x)):
        target = get_max_y(x[i], x, y)
        if y[i][target] == 1:
            new_x.append(x[i])
            new_y.append(y[i])

    return np.array(new_x), np.array(new_y)


def get_max_y(cur_x, x, y):
    y = [y[i] for i, x in enumerate(x) if x == cur_x]

    counts = [0 for i in range(len(y[0]))]

    for i in range(len(y)):
        y[i] = y[i].index(1)

    for i in range(len(y)):
        counts[y[i]] += 1

    return counts.index(max(counts))


def get_class_count(y):
    """
    Description
    -----------
    Get the count for each class in the labels

    Parameters
    ----------
    y : numpy array
        Dataset labels

    Returns
    -------
    The count of each class
    """
    count = [0 for i in range(len(y[0]))]

    for labels in y:
        for index, label in enumerate(labels):
            if label == 1:
                count[index] += 1

    return count


def get_class_indexes(y):
    """
    Description
    -----------
    Get the indexes of each class in the labels

    Parameters
    ----------
    y : numpy array
        Dataset labels

    Returns
    -------
    The indexes of each class
    """
    indexes = [[] for i in range(len(y[0]))]
    for i, labels in enumerate(y):
        for index, label in enumerate(labels):
            if label == 1:
                indexes[index].append(i)

    return indexes


def predict(x, l_lists):
    """
    Description
    -----------
    Get the indexes of each class in the labels

    Parameters
    ----------
    x : numpy array
        Dataset features

    l_lists : list
        set of scoring systems

    Returns
    -------
    The predictions of the set of scoring systems for the given entries
    """
    y = []

    for _, sample in enumerate(x):
        scores = [sum(feature * l_list[j] for j, feature in enumerate(sample)) for l_list in l_lists]

        y_pred = []
        for i in range(len(scores)):
            if i == np.argmax(scores):
                y_pred.append(1)
            else:
                y_pred.append(0)
        y.append(y_pred)

    return np.array(y)


def format_labels(y):
    y_formatted = [i for labels in y for i in range(len(labels)) if labels[i] == 1]
    return y_formatted


def get_accuracy(x, y, l_lists):
    """
    Description
    -----------
    Compute accuracy for the scoring systems

    Parameters
    ----------
    x : numpy array
        Dataset features

    y : numpy array
        Dataset labels

    l_lists : list
        set of scoring systems

    Returns
    -------
    The accuracy of the predictions
    """
    y_pred = predict(x, l_lists)
    y = format_labels(y)
    y_pred = format_labels(y_pred)
    accuracy = accuracy_score(y, y_pred)

    return accuracy


def get_balanced_accuracy(x, y, l_lists):
    y_pred = predict(x, l_lists)
    y = format_labels(y)
    y_pred = format_labels(y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)

    return balanced_accuracy
