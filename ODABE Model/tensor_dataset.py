import torch
import torch.utils.data as data


class TensorDataset(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        t = self.dataset[i]
        return TensorDataset.map(t, TensorDataset.to_tensor)