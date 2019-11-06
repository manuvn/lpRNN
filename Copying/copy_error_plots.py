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
sns.set()#font_scale=1.2)
sns.set_style("ticks")

colors = mcolors.CSS4_COLORS
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
by_hsv = sorted(by_hsv, reverse=True)
color_names = [name for hsv, name in by_hsv]
print(f'Number of colors = {len(color_names)}')




if __name__ == '__main__':
    # python copy_error_plots.py --file_name=lpRNN_H256_HL1_good
    # python copy_error_plots.py --file_name=LSTM_H128_HL1_good
    # python copy_error_plots.py --file_name=lpLSTM_H128_HL1_good
    path = './results/copying/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='LSTM_H128_HL1_128_1LSTM.log',
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
    print(logs[0])

    fig, ax = plt.subplots()
    for k in logs.keys():
        error, T, S, K = logs[k]
        ylen = len(error)
        # ytick = np.linspace(idx,idx+ylen,ylen)
        ax.axvspan(idx, idx+ylen, alpha=0.4, color=color_names[coloridx], lw=0)
        if (old_length != T) & (old_length != 0):
            coloridx += 1
            xticks_idx = idx - block_counter/2 + 1
            xticks_list = xticks_list + [xticks_idx]
            xticks_labels = xticks_labels + [old_length]
            block_counter = 0
        block_counter += ylen
        idx = idx + ylen
        e = np.concatenate((e, error))
        old_length = T

    xticks_idx = idx - block_counter/2
    xticks_list = xticks_list + [xticks_idx]
    xticks_labels = xticks_labels + [old_length]

    # ax.set_xticks(xticks_list)
    # ax.set_xticklabels(xticks_labels)
    # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    # plt.title(r'Copy task (S=8, K=8) with T $\in [3,200]$', weight='bold')
    plt.title(r'Copy task (S=8, T=8) with T $\in [3,500]$', weight='bold')
    plt.plot(e)
    plt.xlabel(r'Epoch', weight='bold')
    plt.ylabel(r'Categorical cross entropy loss', weight='bold')

    ax2 = ax.twiny()
    ax2.set_xticks(xticks_list)
    ax2.set_xticklabels(xticks_labels)
    plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')

    # plt.xlabel(r'Length of the copy task')

    plt.savefig(f'{args.file_name}.pdf',  bbox_inches="tight", layout='tight')
    plt.show()