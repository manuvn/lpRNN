# -*- coding: utf-8 -*-
"""LSTM The Adding Problem dataset gen functions

Original Author: Yuhuang Hu
Modified by: Manu V Nair
email - mnair@ini.uzh.ch
"""

from __future__ import print_function
import os
import pickle
import keras
import matplotlib.pyplot as plt
import numpy as np

#%%
def generate_copying_problem(num_samples=1, Tlen=100,Slen=20, Klen=6, NClasses=7):
    """
    ALERT: Using this function can be dangerous for the task if 
            NCLasses and Klen are not properly chosen
    Klen: Number of symbols to be remembered
    Slen: Length of the data sequence to be remembered
    Tlen: Length of the memory of the task
    NClasses: Should be at least Klen + 1
              Is the seperation symbol. Separation symbol is NClasses - 1
    """
    sample_len = Tlen + 2*Slen
    X = np.zeros((num_samples, sample_len, 1))
    data = np.random.randint(low = 1, high = Klen, size = (num_samples, Slen, 1))
    X[:, :Slen] = data
    X[:, Slen+Tlen] = NClasses-1
    X_data = keras.utils.to_categorical(X, num_classes=NClasses)
    Y = np.zeros((num_samples, sample_len, 1))
    Y[:, -Slen:] = X[:, :Slen]
    Y_data = keras.utils.to_categorical(Y, num_classes=NClasses)
    
    # X   = np.argmax(X_data,axis=-1) 
    # plt.figure()
    # plt.plot(X[0,:],alpha=0.6)
    # X   = np.argmax(Y_data,axis=-1) 
    # plt.plot(X[0,:],alpha=0.6)
    # plt.show()
    return X_data, Y_data

#%%
def load_copying_problem(Ntrain=10000, Ntest=1000, Tlen=10, Slen=10, Klen=10):
    """Loading The Copying Problem Dataset.

    Parameters
    ----------
    mode : str
        add-150, add-200, add-300, add-400
    """
    ds_fn = "copy_problem_len_T"+str(Tlen)+'_S'+str(Slen)+'_K'+str(Klen)
    ds_fn = os.path.join('./support/data/', ds_fn)

    if not os.path.isfile(ds_fn):
        #raise ValueError("The file %s does not exist." % (ds_fn))
        print("The file %s does not exist." % (ds_fn))
        print('So generating myself ... sigh.') 
        X,Y = generate_copying_problem(num_samples=Ntrain, Tlen=Tlen,Slen=Slen, Klen=Klen)
        X_test,Y_test = generate_copying_problem(num_samples=Ntest, Tlen=Tlen, Slen=Slen, Klen=Klen)
        
        with open(ds_fn, "wb") as f:
            pickle.dump((X,Y,X_test,Y_test),f,pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(ds_fn, "rb") as f:
            X, Y, X_test, Y_test = pickle.load(f)

    return X, Y, X_test, Y_test


if __name__ == '__main__':
    generate_copying_problem(Klen=10, NClasses=12)