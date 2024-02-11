import torch.utils.data as data

class GazeCaptureDataset(data.Dataset):

    def __init__(self, df):
        self.df = df
