from argparse import ArgumentParser
from os import listdir

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_load import get_gc_datasets, get_custom_datasets
from model_vgg import NetVgg

device = torch.device('cuda')
screen_width = 1920
screen_height = 1080


def loss_fn(x_true, y_true, x_pred, y_pred):
    return torch.mean(torch.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2))


def plot_and_save_debug_image(i, face_crop, x_true, y_true, x_pred, y_pred, loss):
    img = np.zeros(shape=(screen_height, screen_width, 3))
    face_crop *= 255
    face_crop = face_crop.astype(np.uint8)

    # picture in picture bottom right margin 10
    img[815:1070, 1655:1910, :] = face_crop
    x_true = int(x_true * screen_width)
    x_pred = int(x_pred * screen_width)
    y_true = int(y_true * screen_height)
    y_pred = int(y_pred * screen_height)
    img = cv2.circle(img, center=(x_true, y_true), radius=100, color=(0, 255, 0), thickness=2)
    img = cv2.circle(img, center=(x_pred, y_pred), radius=100, color=(255, 0, 0), thickness=2)
    img = cv2.putText(img, '{:.2f}'.format(loss), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2, cv2.LINE_AA)
    cv2.imwrite('simulation/{}.jpg'.format(i), img)
    print()