import numpy as np
import torch  # lgtm [py/repeated-import] lgtm [py/import-and-import-from]
from torch import nn  # lgtm [py/repeated-import]
from torch import optim  # lgtm [py/repeated-import]
from torch.utils.data import DataLoader  # lgtm [py/repeated-import]
from torch.utils.data.dataset import Dataset

from holisticai.robustness.mitigation.utils.formatting import from_cuda, to_cuda


class MembershipInferenceAttackModel(nn.Module):
    """
    Implementation of a pytorch model for learning a membership inference attack.

    The features used are probabilities/logits or losses for the attack training data along with
    its true labels.
    """

    def __init__(self, num_classes, num_features=None):

        self.num_classes = num_classes
        if num_features:
            self.num_features = num_features
        else:
            self.num_features = num_classes

        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x_1, label):
        """Forward the model."""
        out_x1 = self.features(x_1)
        out_l = self.labels(label)
        is_member = self.combine(torch.cat((out_x1, out_l), 1))
        return self.output(is_member)


class AttackDataset(Dataset):
    """
    Implementation of a pytorch dataset for membership inference attack.

    The features are probabilities/logits or losses for the attack training data (`x_1`) along with
    its true labels (`x_2`). The labels (`y`) are a boolean representing whether this is a member.
    """

    def __init__(self, x_1, x_2, y=None):
        import torch  # lgtm [py/repeated-import] lgtm [py/import-and-import-from]

        self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(torch.FloatTensor)
        self.x_2 = torch.from_numpy(x_2.astype(np.int32)).type(torch.FloatTensor)

        if y is not None:
            self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
        else:
            self.y = torch.zeros(x_1.shape[0])

    def __len__(self):
        return len(self.x_1)

    def __getitem__(self, idx):
        if idx >= len(self.x_1):  # pragma: no cover
            raise IndexError("Invalid Index")

        return self.x_1[idx], self.x_2[idx], self.y[idx]


class Trainer:
    def __init__(self, learning_rate, epochs, batch_size, attack_model):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.attack_model = attack_model
        self.batch_size = batch_size

    def train(self, dataset):
        train_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate)  # type: ignore

        self.attack_model = to_cuda(self.attack_model)  # type: ignore
        self.attack_model.train()  # type: ignore

        for _ in range(self.epochs):
            for (input1, input2, targets) in train_loader:
                input1, input2, targets = (
                    to_cuda(input1),
                    to_cuda(input2),
                    to_cuda(targets),
                )
                _, input2 = torch.autograd.Variable(input1), torch.autograd.Variable(
                    input2
                )
                targets = torch.autograd.Variable(targets)

                optimizer.zero_grad()
                outputs = self.attack_model(input1, input2)  # type: ignore
                loss = loss_fn(
                    outputs, targets.unsqueeze(1)
                )  # lgtm [py/call-to-non-callable]

                loss.backward()
                optimizer.step()

    def predict(self, dataset):
        test_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.attack_model.eval()  # type: ignore
        predictions = []
        for input1, input2, _ in test_loader:
            input1, input2 = to_cuda(input1), to_cuda(input2)
            predicted = self.attack_model(input1, input2)  # type: ignore
            predicted = from_cuda(predicted)
            predictions.append(predicted.detach().numpy())

        predictions = np.vstack(predictions)

        return predictions
