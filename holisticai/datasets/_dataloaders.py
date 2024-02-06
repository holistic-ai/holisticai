# Base Imports
from os import environ, makedirs
from os.path import expanduser, join

# sklearn imports
from sklearn.datasets import fetch_openml


def get_data_home(data_home=None):
    """
    Return the path of the holisticai data directory.
    By default the data directory is set to a folder named 'holisticai_data' in the
    user home folder.
    Alternatively, it can be set by the 'HOLISTICAI_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str, default=None
        The path to holisticai data directory. If `None`, the default path
        is `~/holisticai_data`.
    Returns
    -------
    data_home: str
        The path to holisticai data directory.
    """
    if data_home is None:
        data_home = environ.get("HOLISTIC_AI_DATA", join("~", "holisticai_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def load_student(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Student Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Student dataset can be obtained by printing
    load_student()['DESCR'].

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we dowload
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1797)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="UCI-student-performance-mat",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_adult(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Adult Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Adult dataset can be obtained by printing
    load_adult()['DESCR'].

    #

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1797)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """

    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="adult",
        version=2,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_law_school(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Law School Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Law School dataset can be obtained by printing
    load_law_school()['DESCR'].

    # Features: 11
    # Instances: 20800
    # Target: 1 = Admitted, 0 = Not Admitted

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1797)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="law-school-admission-bianry",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_last_fm(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Last FM Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Last FM dataset can be obtained by printing
    load_last_fm()['DESCR'].

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1797)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="LastFM_dataset",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_us_crime(data_home=None, return_X_y=False, as_frame=True):
    """
    Load US Crime Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the US Crime dataset can be obtained by printing
    load_us_crime()['DESCR'].

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="us_crime",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_heart(data_home=None, return_X_y=False, as_frame=True):
    """
    Load heart Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the heart dataset can be obtained by printing
    load_us_crime()['DESCR'] or in its original repository
    https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records.

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="heart-failure",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_german_credit(data_home=None, return_X_y=False, as_frame=True):
    """
    Load german_credit Dataset.

    Description
    ----------
    This dataset classifies people described by a set of attributes as good or bad credit risks.

    # Features: 24
    # Instances: 1000
    # Target: 1 = Good, 2 = Bad

    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="credit-g",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_census_kdd(data_home=None, return_X_y=False, as_frame=True):
    """
    Load census_kdd Dataset.

    Description
    ----------
    This data set contains weighted census data extracted from the 1994 and 1995
    Current Population Surveys conducted by the U.S. Census Bureau. The data contains
    41 demographic and employment related variables. The instance weight indicates the
    number of people in the population that each record represents due to stratified sampling.
    To do real analysis and derive conclusions, this field must be used. This attribute should
    not be used in the classifiers. One instance per line with comma delimited fields.

    # Features: 31
    # Instances: 299285
    # Target: income

    Source: https://archive.ics.uci.edu/dataset/117/census+income+kdd

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="Census-Income-KDD",
        version=3,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_bank_marketing(data_home=None, return_X_y=False, as_frame=True):
    """
    Load bank_marketing Dataset.

    Description
    ----------
    The data is related with direct marketing campaigns of a Portuguese banking institution.
    The marketing campaigns were based on phone calls. Often, more than one contact to the same
    client was required, in order to access if the product (bank term deposit) would be (or not) subscribed.
    The classification goal is to predict if the client will subscribe a term deposit (variable y).

    # Features: 16
    # Instances: 45211
    # Target: has the client subscribed a term deposit? yes/no

    Source: https://archive.ics.uci.edu/ml/datasets/bank+marketing

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="bank-marketing",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_credit_card(data_home=None, return_X_y=False, as_frame=True):
    """
    Load credit_card Dataset.

    Description
    ----------
    This research aimed at the case of customersaEUR(tm) default payments in Taiwan
    and compares the predictive accuracy of probability of default among six data mining methods.

    # Features: 23
    # Instances: 30000
    # Target: default payment next month (yes/no)

    Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="default-of-credit-card-clients",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_compas_recidivism(data_home=None, return_X_y=False, as_frame=True):
    """
    Load compass_recidivism Dataset.

    Description
    ----------
    The data was subsequently preprocessed and reduced to relevant features for classification.
    The target variable is two_year_recid which indicates recidivism.

    # Features: 13
    # Instances: 5278
    # Target: two_year_recid

    Source: https://github.com/propublica/compas-analysis/

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="compas-two-years",
        version=5,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch


def load_diabetes(data_home=None, return_X_y=False, as_frame=True):
    """
    Load diabetes Dataset.

    Description
    ----------
    he dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and
    integrated delivery networks. Each row concerns hospital records of patients diagnosed with diabetes,
    who underwent laboratory, medications, and stayed up to 14 days.
    The goal is to determine the early readmission of the patient within 30 days of discharge.

    # Features: 47
    # Instances: 101766
    # Target: readmitted

    Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

    Parameters
    ----------
    data_home : str
        The directory to which the data is downloaded. If None, we download
        to the holisticai_data folder in user home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1994, 127)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1994,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        frame: DataFrame of shape (1994, 128)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        (data, target) : tuple if ``return_X_y`` is True
            A tuple of two ndarrays by default. The first contains a 2D ndarray of
            shape (num_rows, num_columns) with each row representing one sample and
            each column representing the features. The second ndarray of shape (1994)
            contains the target samples.  If `as_frame=True`, both arrays are pandas
            objects, i.e. `X` a dataframe and `y` a series.
    """
    if data_home is None:
        data_home = get_data_home()

    bunch = fetch_openml(
        name="Diabetes130US",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )

    return bunch
