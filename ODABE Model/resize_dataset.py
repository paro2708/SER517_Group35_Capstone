import torch.utils.data as data

class ResizeDataset(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
