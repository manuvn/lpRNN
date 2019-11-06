# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def plot_history(path,rnns,histories,loss_label,acc_label, loss_type, acc_type):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.despine()
    idx = 0
    for hist in histories:
        plt.plot(hist.history[loss_type],label=rnns[idx])
        idx = idx + 1
    plt.ylabel(loss_label)
    plt.xlabel('Iterations')
    plt.legend()   

    idx = 0
    plt.savefig(path+'loss.pdf')
    plt.figure()
    for hist in histories:
        plt.plot(hist.history[acc_type],label=rnns[idx])
        idx = idx + 1
    plt.ylabel(acc_label)
    plt.xlabel('Epochs')
    plt.legend()        
    plt.savefig(path+'acc.pdf')
    
def gen_babi_table():
    fpath = '../results/bAbi/1k'
    with open(fpath, "rb") as f:
        results = pickle.load(f)
    return results
        