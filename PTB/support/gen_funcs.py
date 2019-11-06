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

import numpy as np
#%%
def adding_problem_generator(N, seq_len=6, high=1):
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
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y

#%%
def load_adding_problem(mode="add-200"):
    """Loading The Adding Problem Dataset.

    Parameters
    ----------
    mode : str
        add-150, add-200, add-300, add-400
    """
    ds_fn = "add_problem_len_"
    if mode == "add-50":
        ds_fn += "50.pkl"
    elif mode == "add-150":
        ds_fn += "150.pkl"
    elif mode == "add-200":
        ds_fn += "200.pkl"
    elif mode == "add-300":
        ds_fn += "300.pkl"
    elif mode == "add-400":
        ds_fn += "400.pkl"
    else:
        raise ValueError("Not a valid mode for the adding problem")

    ds_fn = os.path.join('./support/data/', ds_fn)

    if not os.path.isfile(ds_fn):
        #raise ValueError("The file %s does not exist." % (ds_fn))
        print("The file %s does not exist." % (ds_fn))
        print('So generating myself ... sigh.')
        if mode == "add-50":
            length = 50
        elif mode == "add-150":
            length = 150
        elif mode == "add-200":
            length = 200
        elif mode == "add-300":
            length = 300
        elif mode == "add-400":
            length = 400
        N_train = 100000
        N_test = 10000
        X,Y = adding_problem_generator(N_train, length)
        X_test,Y_test = adding_problem_generator(N_test, length)
        
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
    x_train = np.reshape(x_train, (-1,28*28,1))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    samples_size = x_train.shape[0]
    while(1):
        indices = np.random.randint(0,samples_size,num_samples)
        x_data = x_train[indices,:,:]
        y_data = y_train[indices,:]
        yield x_data, y_data

#%%
def generate_copying_problem(num_samples=100000, Tlen=100,Slen=20, Klen=3):

    sample_len = Tlen + 2*Slen
    X = np.zeros((num_samples, sample_len, 1))
    data = np.random.randint(low = 1, high = Klen, size = (num_samples, Slen, 1))
    # blank category = 0
    # sepration symbol = 9
    X[:, :Slen] = data
    X[:, Slen+Tlen] = Klen
    X_data = keras.utils.to_categorical(X, num_classes=Klen+1)
    Y = np.zeros((num_samples, sample_len, 1))
    Y[:, -Slen:] = X[:, :Slen]
    Y_data = keras.utils.to_categorical(Y, num_classes=Klen+1)
    
    X   = np.argmax(X_data,axis=-1) 
#    plt.figure()
#    plt.plot(X[0,:],'r',alpha=0.6)
#    plt.show()
#    print(X[0:-Slen])    
#    X   = np.argmax(Y_data,axis=-1) 
#    plt.figure()
#    plt.plot(X[0,:],'r',alpha=0.6)
#    plt.show()
#    print(X[0:-Slen])
    return X_data, Y_data

#%%
def load_copying_problem(Tlen=10, Slen=10, Klen=10):
    """Loading The Adding Problem Dataset.

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
 
        N_train = 50000
        N_test = 10000
        X,Y = generate_copying_problem(num_samples=N_train, Tlen=Tlen,Slen=Slen, Klen=Klen)
        X_test,Y_test = generate_copying_problem(num_samples=N_test, Tlen=Tlen, Slen=Slen, Klen=Klen)
        
        with open(ds_fn, "wb") as f:
            pickle.dump((X,Y,X_test,Y_test),f,pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(ds_fn, "rb") as f:
            X, Y, X_test, Y_test = pickle.load(f)

    return X, Y, X_test, Y_test