import torch.utils.data as data

class GcNormalizingDataset(data.Dataset):

    def __init__(self, dataset, mean, stddev):
        self.dataset = dataset
        self.mean = mean
        self.stddev = stddev
