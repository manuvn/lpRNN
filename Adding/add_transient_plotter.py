# -*- coding: utf-8 -*-

""" 
The Adding task plotter support code
"""
from __future__ import print_function
import os
import pickle
import argparse

# to force CPU - Appears to be much faster than GPU (not using cuDNN)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from keras import models

import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Lambda
from keras.layers import Dense, Reshape
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop, SGD
# from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from lprnn_keras_impl import lpRNN, PlpRNN, IndRNN, lpIndRNN, define_model, lpLSTM, eLSTM, mlpGRU
from support import load_adding_problem, AccHistory, LossHistory, plot_history
from support import info, debug, printer

from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, ModelCheckpoint

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import seaborn as sns
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

import seaborn as sns
sns.set()
sns.set_style("whitegrid")

def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
    """
    x = x.astype('float32')
    y = np.clip(x * 0.2 + 0.5, 0, 1)
    return y.astype('float32')

def relu(x):
    """
    Relu function
    """
    # print('x',x.dtype)
    x = x.astype('float32')
    y = np.maximum(x, 0, x)
    y = y.astype('float32')
    return y

def tanh(x):
    """
    Tanh function
    """
    x = x.astype('float32')
    y = np.tanh(x)
    y = y.astype('float32')
    return y#np.tanh(x)

def add_task(RNN_model, model_name='RNN', activation='tanh',
             length=400, nb_layers=1, nb_hid=128,
             initializer_func=None, hybrid_nunits=0,
             hybrid_type=LSTM, recurrent_initializer='orthogonal',
             load_model=False, N_train=1, 
             N_test=1, model_path=None,max_entries=3):
    """Perform the adding task

    Parameters
    ----------
    RNN_model : handle to the class
    model_name: Just a name
    length : int
    nb_layers : int
        number of recurrent layers
    nb_hid : int
        number of hidden units per layer
    nb_epoch : int
        total number of epoch
    batch_size : int
        the batch size
    learning_rate : float
        learning rate of RMSprop optimizer
    clipnorm : float
        gradient clipping, if >0
    """
#    model_path = os.path.join('./results/masked_addition/',
#                              model_name + '_' + str(nb_hid) + '_' + str(nb_layers) + '.h5')
    model_pic = os.path.join(model_path, model_name + "-model-pic.png")

    # ----- print mode info -----
    info("Model Name: ", model_name)
    info("Number of layers: ", nb_layers)
    info("Number of hidden units: ", nb_hid)
    info("Activation: ", activation)
    info("Recurrent initializer: ", recurrent_initializer)
    
    # ----- prepare data -----
    # identify data format
    if K.backend() == "tensorflow":
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")
    data_format = K.image_data_format()

    X_train, Y_train, _, _ = load_adding_problem(length=length,  
                                                N_train=N_train, 
                                                N_test=N_test, 
                                                max_entries=max_entries, 
                                                save=False,
                                                load=False)

    info("Basic dataset statistics")
    info("X_train shape:", X_train.shape)
    info("Y_train shape:", Y_train.shape)
    
    # setup sequence shape
    input_shape = X_train.shape[1:]

    # ----- Build Model -----
    img_input = Input(shape=input_shape)

    if initializer_func == None:
        initializer_func = keras.initializers.Identity(gain=1.0)

    x = define_model(RNN_model=RNN_model,
                     hybrid_nunits=hybrid_nunits,
                     h_dim=nb_hid,
                     op_dim=1,
                     num_layers=nb_layers,
                     activation=activation,
                     op_activation='linear',
                     recurrent_initializer=recurrent_initializer,
                     ip=img_input,
                     GRNN=hybrid_type,
                     op_type='sample',
                     learn_retention_ratio=True)

    # compile model
    info('Compiling model...')
    model = Model(img_input, x)
    model.summary()
    
    if not os.path.isfile(model_path):
        debug('ALERT - Model file does not exist. Ending.', model_path)
        return
    else:
        model.load_weights(model_path, by_name=True)


    # ---- Record activations -----
    info('---------------------- Collecting the activations -------------------')
    layer_outputs = [layer.output for layer in model.layers[1:]] # 0 is the input layer
    activation_model = Model(img_input, layer_outputs)
    activations = activation_model.predict(X_train)

    # printing weights of the network
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    idx = 0
    for name, weight in zip(names, weights):
        print(name, weight.shape, idx)
        idx += 1

    idx = 0

    if (RNN_model == LSTM):
        print('ALERT: Makes assumptions about format. This may break!')
        kernel = weights[0]
        recurrent_kernel = weights[1]
        bias = weights[2]
        dense_kernel = weights[3]
        dense_bias = weights[4]

        # ------- Plotting gates ------
        units = nb_hid
        kernel_i = kernel[:, :units]
        kernel_f = kernel[:, units: units * 2]
        kernel_c = kernel[:, units * 2: units * 3]
        kernel_o = kernel[:, units * 3:]

        recurrent_kernel_i = recurrent_kernel[:, :units]
        recurrent_kernel_f = recurrent_kernel[:, units: units * 2]
        recurrent_kernel_c = recurrent_kernel[:, units * 2: units * 3]
        recurrent_kernel_o = recurrent_kernel[:, units * 3:]

        bias_i = bias[:units]
        bias_f = bias[units: units * 2]
        bias_c = bias[units * 2: units * 3]
        bias_o = bias[units * 3:]

        activation = eval(activation)
        ractivation = hard_sigmoid
        h_tm1 = np.zeros((1,units), dtype='float32')
        c_tm1 = np.zeros((1,units), dtype='float32')
        op = np.zeros(length, dtype='float32')
        X_train = X_train.astype('float32')
        fg = np.zeros((units, length))
        ig = np.zeros((units, length))
        og = np.zeros((units, length))
        c_log = np.zeros((units, length))
        h_log = np.zeros((units, length))
        for idx in range(length):
            inputs = X_train[:,idx,:]

            x_i = np.dot(inputs, kernel_i)
            x_f = np.dot(inputs, kernel_f)
            x_c = np.dot(inputs, kernel_c)
            x_o = np.dot(inputs, kernel_o)

            x_i = np.add(x_i, bias_i)
            x_f = np.add(x_f, bias_f)
            x_c = np.add(x_c, bias_c)
            x_o = np.add(x_o, bias_o)

            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

            i = ractivation(np.add(x_i, np.dot(h_tm1_i, recurrent_kernel_i)))
            f = ractivation(np.add(x_f, np.dot(h_tm1_f, recurrent_kernel_f)))

            c = np.add(f * c_tm1, i * activation(np.add(x_c, np.dot(h_tm1_c, recurrent_kernel_c))))

            o = ractivation(np.add(x_o, np.dot(h_tm1_o, recurrent_kernel_o)))
            h = o * activation(c)

            h_tm1 = h + 0
            c_tm1 = c + 0
            fg[:,idx], ig[:,idx], og[:,idx], c_log[:,idx], h_log[:,idx] = f, i, o, c, h

        idx = 1
        # for idx in range(units):
        plt.plot(fg[idx,:], label='Forget gate')
        plt.plot(ig[idx,:], label='Input gate')
        plt.plot(og[idx,:], label='Output gate')
        plt.plot(c_log[idx,:], label='Internal state')
        plt.plot(h_log[idx,:], label='Output')
        legend_properties = {'weight':'bold'}
        plt.plot(X_train[0,:,1],'*', label='Mask')
        plt.legend(prop=legend_properties)
        plt.savefig('LSTM_mask_add.pdf', dpi=500)

        # Dense layer computation
        op = np.dot(h_tm1, dense_kernel)
        op = np.add(op, dense_bias)

    if(RNN_model == lpRNN):
        op = np.zeros(length, dtype='float32')
        rnn_activation = activations[0]
        dense_kernel = weights[4]
        dense_bias = weights[5]
        for idx in range(length):
            op[idx] = np.dot(rnn_activation[0,idx,:],dense_kernel)
            op[idx] += dense_bias
        plt.plot(X_train[0,:,1],'*', label='Mask')
        print(op.shape)
        for idx in range(nb_hid):
            plt.plot(rnn_activation[0,:,idx], alpha=0.6, lw=1)
        plt.plot(op, label='Activation', lw=3)

    plt.xlabel(r'Time step', weight='bold')
    plt.ylabel(r'Value', weight='bold')
    plt.show()

    # ----- Inference run -----
    output = model.predict(X_train)
    info(f'Expected output = {Y_train}. Actual output = {output}')

    return output


if __name__ == '__main__':
    # python add_transient_plotter.py --nb_hid=2 --rnn_model=LSTM --N_train=1 --max_entries=20 --length=100 --fname=LSTM_H0_2_1_nocurr
    # python add_transient_plotter.py --nb_hid=2 --rnn_model=LSTM --N_train=1 --max_entries=30 --length=200000 --fname=LSTM_H0_2_1_curr_good
    path = './results/masked_addition/'
    hist = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_hid', type=int, default=32,
                        help='number of hidden layer units')
    parser.add_argument('--nb_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--max_entries', type=int, default=3,
                        help='number of unmasked entries')
    parser.add_argument('--length', type=int, default=50000,
                        help='Length to be tested')
    parser.add_argument('--rnn_model', type=str, default='lpRNN',
                        help='RNN model type - LSTM, lpRNN, GRU, lpLSTM, IndRNN, lpIndRNN')
    parser.add_argument('--fname', type=str, default='rnn_cell',
                        help='Name of the model file to be saved')
    parser.add_argument('--nhybrid', type=int, default=0,
                        help='Number of hybrid units when mixing lpRNN with LSTMs ')
    parser.add_argument('--hybrid_type', type=str, default='LSTM',
                        help='Hybrid type')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--nb_epoch', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--N_train', type=int, default=10000,
                        help='Training data size')
    parser.add_argument('--N_test', type=int, default=1000,
                        help='Test data size')
                        
    args = parser.parse_args()

    nb_hid      = args.nb_hid #32
    nb_layers   = args.nb_layers #1
    max_entries = args.max_entries #3 
    length      = args.length #10
    nhybrid     = args.nhybrid #0
    hybrid_type = args.hybrid_type #LSTM
    activation  = args.activation #relu
    N_test      = args.N_test # test data length
    N_train     = args.N_train # train length
    rnn_model_name = args.rnn_model #'LSTM'
    fname       = args.fname #relu

    kwargs      = {}
    name = rnn_model_name+str(nb_hid)
    model_path = os.path.join('./results/masked_addition/', fname + '.h5')
    kwargs[rnn_model_name] = {
              'model_path': model_path
              ,'nb_hid': nb_hid
              ,'nb_layers':nb_layers
              ,'hybrid_nunits':nhybrid
              ,'hybrid_type':hybrid_type
              ,'activation':activation
              ,'N_train': N_train 
              ,'N_test': N_test
              }

    rnn_model = eval(rnn_model_name)

    logs = add_task(RNN_model = rnn_model
                    ,model_name=rnn_model_name
                    ,length = length
                    ,max_entries = max_entries
                    ,**kwargs[rnn_model_name])