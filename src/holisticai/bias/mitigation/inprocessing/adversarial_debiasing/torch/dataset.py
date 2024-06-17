import torch
from torch.utils.data import Dataset


class ADDataset(Dataset):
    """
    Dataset class to train adversarial debiasing pytorch model.
    """

    def __init__(self, X, y, groups_num, device):
        self.X = X
        self.y = y
        self.groups_num = groups_num
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx]).type(torch.FloatTensor).to(self.device)
        y = torch.tensor(self.y[idx, None]).type(torch.FloatTensor).to(self.device)
        group = (
            torch.tensor(self.groups_num[idx, None])
            .type(torch.FloatTensor)
            .to(self.device)
        )
        return (X, y), (y, group)
