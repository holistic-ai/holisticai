def get_attack_model(attack_model_type):
    """
    Returns an attack model of the specified type.

    Parameters
    ----------
    attack_model_type : str
        The type of the attack model. Possible values are 'nn' for neural network and 'rf' for random forest.

    Returns
    -------
    object
        The attack model.
    """
    if attack_model_type == "nn":
        from sklearn.neural_network import MLPClassifier

        attack_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=2000,
            shuffle=True,
            random_state=None,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10,
            max_fun=15000,
        )
    elif attack_model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier

        attack_model = RandomForestClassifier()
    else:
        raise ValueError("Illegal value for parameter `attack_model_type`.")
    return attack_model


def is_pipeline(estimator):
    """
    Checks if the given estimator is a pipeline.

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Returns
    -------
    bool
        True if the estimator is a pipeline.
    """
    from holisticai.pipeline import Pipeline

    if type(estimator) is Pipeline:
        return True
    return None


def model_in_pipeline(estimator):
    """
    Returns the model in a pipeline.

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Returns
    -------
    object
        The model in the pipeline.
    """
    return estimator.estimator_hdl.estimator.obj


def is_estimator_valid(estimator, estimator_requirements) -> bool:
    """
    Checks if the given estimator satisfies the requirements for this attack.

    Parameters
    ----------
    estimator : object
        The estimator to check.
    estimator_requirements : list
        The requirements for the estimator.

    Returns
    -------
    bool
        True if the estimator is valid for the attack.
    """

    if is_pipeline(estimator):
        model = model_in_pipeline(estimator)
        return is_estimator_valid(model, estimator_requirements)

    for req in estimator_requirements:
        # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
        # estimator is required to inherit from at least one of the classes
        if isinstance(req, tuple):
            if all(p not in type(estimator).__mro__ for p in req):
                return False
        elif req not in type(estimator).__mro__:
            return False
    return True


def get_feature_index(feature):
    """
    Returns a modified feature index: in case of a slice of size 1, returns the corresponding integer. Otherwise,
    returns the same value (integer or slice) as passed.

    Parameters
    ----------
    feature : int or slice
        The index or slice representing a feature to attack (0-based).

    Returns
    -------
    int or slice
        An integer representing a single column index or a slice representing a multi-column index.
    """
    if isinstance(feature, int):
        return feature

    start = feature.start
    stop = feature.stop
    step = feature.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    if feature.stop is not None and ((stop - start) // step) == 1:
        return start

    return feature
