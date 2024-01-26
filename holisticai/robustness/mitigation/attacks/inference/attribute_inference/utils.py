def get_attack_model(attack_model_type):
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
