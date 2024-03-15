from collections import deque

import cv2
import numpy as np
from os import listdir
from os.path import join, isdir, splitext, isfile

from scipy.io import loadmat

from face_extractor import extract_face


def get_participant_screen_size(participant_dir_fullname):
    screen_size_file = join(participant_dir_fullname, 'Calibration', 'screenSize.mat')
    screen_size = loadmat(screen_size_file)
    screen_w = int(np.squeeze(screen_size['width_pixel']))
    screen_h = int(np.squeeze(screen_size['height_pixel']))
    return screen_w, screen_h