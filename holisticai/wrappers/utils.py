def build_linear_regression_pytorch_model(nb_features, nb_classes):
    import torch

    from holisticai.wrappers.classification import PyTorchClassifier

    class LinearRegression(torch.nn.Module):
        def __init__(self, inputSize, outputSize):
            super(LinearRegression, self).__init__()
            self.linear = torch.nn.Linear(inputSize, outputSize)

        def forward(self, x):
            out = self.linear(x)
            return out

    model = LinearRegression(nb_features, nb_classes)
    loss = torch.nn.MSELoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        nb_classes=nb_classes,
        input_shape=(nb_features,),
        optimizer=optimizer,
    )
    return classifier
