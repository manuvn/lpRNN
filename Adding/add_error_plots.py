import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
import matplotlib as mpl

rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import seaborn as sns
sns.set(font_scale=1.2)
sns.set_style("ticks")


colors = mcolors.CSS4_COLORS
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
by_hsv = sorted(by_hsv, reverse=True)
color_names = [name for hsv, name in by_hsv]
print(f'Number of colors = {len(color_names)}')




if __name__ == '__main__':
    # python add_error_plots.py --file_name=LSTM_H2_logs_s10_m10001_me10_good
    # python add_error_plots.py --file_name=lpRNN_H128_logs_s10_me3_good
    path = './results/masked_addition/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='lpRNN_logs_s10_m50000',
                        help='Masked addition error log')
    args = parser.parse_args()

    path = path + args.file_name
    # open a file, where you stored the pickled data
    file = open(path, 'rb')

    # dump information to that file
    logs = pickle.load(file)

    # close the file
    file.close()

    e = np.array([])
    y = np.array([])
    xticks_list = []
    xticks_labels = []
    idx = 0
    coloridx = 20
    old_length = 0
    block_counter = 0

    fig, ax = plt.subplots()
    for k in logs.keys():
        error, length = logs[k]
        ylen = len(error)
        # ytick = np.linspace(idx,idx+ylen,ylen)
        ax.axvspan(idx, idx+ylen, alpha=0.4, color=color_names[coloridx], lw=0)
        if (old_length != length) & (old_length != 0):
            coloridx += 2
            xticks_idx = idx - block_counter/2 + 1
            xticks_list = xticks_list + [xticks_idx]
            xticks_labels = xticks_labels + [old_length]
            block_counter = 0
        block_counter += ylen
        idx = idx + ylen
        e = np.concatenate((e, error))
        old_length = length

    xticks_idx = idx - block_counter/2
    xticks_list = xticks_list + [xticks_idx]
    xticks_labels = xticks_labels + [old_length]

    plt.plot(e)
    plt.xlabel(r'Epoch', weight='bold')
    plt.ylabel(r'MSE', weight='bold')

    ax2 = ax.twiny()
    ax2.set_xticks(xticks_list)
    ax2.set_xticklabels(xticks_labels)
    plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.xlabel(r'Length of the masked addition task', weight='bold')

    plt.savefig(f'{args.file_name}.pdf',  bbox_inches="tight", layout='tight')
    plt.show()
