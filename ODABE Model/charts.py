import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def smooth(series, alpha=0.9):
    for i in range(1, len(series)):
        series[i] = (1 - alpha) * series[i - 1] + alpha * series[i]


def plot(pre_training, stage1_online, stage2_online, stage1_offline, stage2_offline):
    assert len(stage1_online) == len(stage1_offline)
    assert len(stage2_online) == len(stage2_offline)
    font_size = 40
    sns.set_style('darkgrid')
    plt.xlabel('Iterations', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)

    online = stage1_online + stage2_online
    offline = stage1_offline + stage2_offline