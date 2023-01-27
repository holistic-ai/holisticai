import torch
from torch import nn
from torch.autograd import Variable


class ClassifierModel(nn.Module):
    "Classifier Model, You can change the layer configuration as you wish!"

    def __init__(self, features_dim, hidden_size, keep_prob):
        super().__init__()
        self.hidden1 = nn.Linear(features_dim, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(keep_prob)
        self.out_act = nn.Sigmoid()

    def forward(self, x, trainable=True):
        x = self.hidden1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.hidden2(x)
        p = self.out_act(x)
        if trainable:
            return p, x
        else:
            return p


class AdversarialModel(nn.Module):
    """
    Adversarial Model, proposed by B. H. Zhang et al.

    Reference:
        B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
        Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
        Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self):
        super().__init__()
        self.c = Variable(
            torch.randn(
                1,
            ).type(torch.FloatTensor),
            requires_grad=True,
        )
        self.act = nn.Sigmoid()
        self.hidden = nn.Linear(3, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x, y, trainable=True):
        s = self.act((1 + torch.absolute(self.c)) * x)
        x = torch.cat((s, s * y, s * (1.0 - y)), 1)
        x = self.hidden(x)
        p = self.out_act(x)
        if trainable:
            return p, x
        else:
            return p


class ADModel(nn.Module):
    """
    Complete system integrate classifier and adversarial submodels.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.classifier = estimator
        self.adversarial = AdversarialModel()

    def forward(self, x, y):
        y_prob, y_logits = self.classifier(x)
        z_prob, _ = self.adversarial(y_logits, y)
        return y_prob, z_prob
