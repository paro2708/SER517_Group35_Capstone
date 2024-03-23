from argparse import ArgumentParser

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, dest='index', help='index of environment dataset')
    args = parser.parse_args()