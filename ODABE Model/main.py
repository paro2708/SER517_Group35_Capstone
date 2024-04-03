import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import settings
from data_load import load_mpii_dataframes
from device_dataset import DeviceDataset
from model import Net
from gazecapture_dataset import MpiiDataset
from gc_normalizing_dataset import MpiiNormalizingDataset
from resize_dataset import ResizeDataset
from tensor_dataset import TensorDataset


def train(data_frame, params):
    device = torch.device('cuda')

    dataset = MpiiDataset(data_frame)
    # dataset = ResizeDataset(dataset)
    dataset = MpiiNormalizingDataset(dataset,
                                     mean=(0.33320256,
                                           0.35879958,
                                           0.45563497),
                                     stddev=(np.sqrt(0.05785664),
                                             np.sqrt(0.06049888),
                                             np.sqrt(0.07370879)))
    dataset = TensorDataset(dataset)
    dataset = DeviceDataset(dataset, device)

    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE)

    model = Net().to(device)

    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)