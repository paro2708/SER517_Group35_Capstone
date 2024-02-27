import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from resize_dataset import ResizeDataset
from tensor_dataset import TensorDataset


def load_mpii_dataframes():
    df = pd.read_csv('data/gazecapture_prepared/metadata.csv')
    df = df.sample(frac=1, random_state=1)
    df_train_valid, df_test = train_test_split(df, random_state=1, test_size=0.15)
    df_train, df_valid = train_test_split(df_train_valid, random_state=1, test_size=0.1765)
    return df_train, df_valid, df_test


def load_custom_dataframes(n, split_index, shuffle=False):
    df = pd.read_csv('data/custom{}_prepared/metadata.csv'.format(n))
    if shuffle:
        df = df.sample(frac=1)
    df_train = df[:split_index]
    df_valid = df[split_index:]
    return df_train, df_valid