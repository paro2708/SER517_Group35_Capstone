import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from data_load import load_mpii_dataframes

if __name__ == '__main__':
    df_train, _, _ = load_mpii_dataframes()

    r_scaler = StandardScaler()
    g_scaler = StandardScaler()
    b_scaler = StandardScaler()