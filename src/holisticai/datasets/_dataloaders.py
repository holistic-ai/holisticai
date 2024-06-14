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

    return fetch_openml(
        name="UCI-student-performance-mat",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def load_adult(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Adult Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Adult dataset can be obtained by printing
    load_adult()['DESCR'].

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

    return fetch_openml(
        name="adult",
        version=2,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def load_law_school(data_home=None, return_X_y=False, as_frame=True):
    """
    Load Law School Dataset.

    Description
    ----------
    We use the sklearn.datasets fetch_openml function to
    fetch data from the openml api. Information about
    the Law School dataset can be obtained by printing
    load_law_school()['DESCR'].

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

    return fetch_openml(
        name="law-school-admission-bianry",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


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

    return fetch_openml(
        name="LastFM_dataset",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


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

    return fetch_openml(
        name="us_crime",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


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

    return fetch_openml(
        name="heart-failure",
        version=1,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )
