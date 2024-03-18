from os import listdir
from os.path import join
from random import choice

import cv2
import numpy as np


def create_array(rows, cols):
    return np.zeros(shape=(rows * 100, cols * 100, 3), dtype=np.uint8)


def set_image(arr, row, col, img):
    l_row = row * 100
    h_row = l_row + 100
    l_col = col * 100
    h_col = l_col + 100
    arr[l_row:h_row, l_col:h_col] = img