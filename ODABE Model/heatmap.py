from argparse import ArgumentParser

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('mtd.csv')
    x = df['x'].values
    y = df['y'].values
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()