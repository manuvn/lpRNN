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
def adding_problem_generator(N, seq_len=6, high=1, max_entries=3):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey
    E. Hintan's paper, A Simple Way to Initialize Recurrent Networks of
    Rectified Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform
                 distribution.
    :return: (X, Y), X the data, Y the label.
    """
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        num_entries = np.random.randint(2,max_entries,1)
        #print(f'Generating data with {num_entries} unmasked locations')
        positions = np.random.choice(seq_len, size=num_entries, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y

#%%
def load_adding_problem(length=200, N_train=20000, N_test=10000, max_entries=3, save=True,load=True):
    """Loading The Adding Problem Dataset.

    Parameters
    ----------
    mode : str
        add-150, add-200, add-300, add-400
    """
    ds_fn = "add_problem_len_" + str(length) + ".pkl"

    ds_fn = os.path.join('./support/data/', ds_fn)

    if (not os.path.isfile(ds_fn)) or (load==False):
        #raise ValueError("The file %s does not exist." % (ds_fn))
        print("The file %s does not exist." % (ds_fn))
        print('So generating myself ... sigh.')           
        X,Y = adding_problem_generator(N_train, length, max_entries=max_entries)
        X_test,Y_test = adding_problem_generator(N_test, length, max_entries=max_entries)
        if (save):
            with open(ds_fn, "wb") as f:
                pickle.dump((X,Y,X_test,Y_test),f,pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(ds_fn, "rb") as f:
            X, Y, X_test, Y_test = pickle.load(f)

    return X, Y, X_test, Y_test

#%%
def generate_mnist_data(num_samples=10):
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_nz = x_train > 0
    x_train = np.reshape(x_train, (-1,28*28,1))
#    x_train = np.concatenate((x_train, x_train_nz),2)
#    x_train = np.reshape(x_train, (-1,28*28,2))
    
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    samples_size = x_train.shape[0]
    while(1):
        indices = np.random.randint(0,samples_size,num_samples)
        x_data = x_train[indices,:,:]
        y_data = y_train[indices,:]
        yield x_data, y_data

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