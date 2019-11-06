# -*- coding: utf-8 -*-

""" 
The Adding task with curriculum learning
"""
from __future__ import print_function
import os
import pickle
import argparse

# to force CPU - Appears to be much faster than GPU (not using cuDNN)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Lambda
from keras.layers import Dense, Reshape
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop, SGD
# from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from lprnn_keras_impl import define_model, lpLSTM, lpRNN
from support import load_adding_problem, AccHistory, LossHistory
from support import info, debug, printer

from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, ModelCheckpoint
from keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt
import math

class StopOnLoss(Callback):
    def __init__(self, monitor='loss', value=1e-5, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs={}):
        self.model.stop_training = False


    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                printer(f"Epoch {epoch}: early stopping loss is {current}, Target is {self.value}")
            self.model.stop_training = True
        
        if math.isnan(current):
            if self.verbose > 0:
                printer("NaN loss. Killing.")
            self.model.stop_training = True

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                 save_best_only=save_best_only, save_weights_only=save_weights_only,
                 mode=mode, period=period)
        
        self.best = 0.05
    def on_epoch_end(self, epoch, logs=None):
        super(MyModelCheckpoint, self).on_epoch_end(epoch=epoch, logs=logs)
            
def add_task(RNN_model, model_name='RNN', activation='tanh',
             length=400, nb_layers=1, nb_hid=128,
             nb_epoch=10, batch_size=32, learning_rate=0.01,
             clipnorm=1000, initializer_func=None, hybrid_nunits=0,
             hybrid_type=LSTM, recurrent_initializer='orthogonal', gate_regularizer=None,
             learn_retention_ratio=False, load_model=False, N_train=20000, 
             N_test=1000, model_path=None,max_entries=3):
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
    info("Number of epochs: ", nb_epoch)
    info("Batch Size: ", batch_size)
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

    X_train, Y_train, X_test, Y_test = load_adding_problem(length=length,  N_train=N_train, N_test=N_test, max_entries=max_entries)

    info("Basic dataset statistics")
    info("X_train shape:", X_train.shape)
    info("Y_train shape:", Y_train.shape)
    info("X_test shape:", X_test.shape)
    info('Y_test shape:', Y_test.shape)
    
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
                     learn_retention_ratio=learn_retention_ratio,
                     gate_regularizer=gate_regularizer)

    # compile model
    info('Compiling model...')
    model = Model(img_input, x)
    model.summary()
    
    if not os.path.isfile(model_path):
        if load_model == True:
            debug('File does not exist. Creating new.')
    else:
        model.load_weights(model_path, by_name=True)

    # ----- Configure Optimizer -----
#    opt = RMSprop(lr=learning_rate, clipnorm=clipnorm)
    opt = SGD(lr=learning_rate, clipnorm=clipnorm)
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mse'])

    print("[MESSAGE] Model is compiled.")

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=25, verbose=1)
    print_model_name = LambdaCallback(on_epoch_begin=lambda batch, logs: info(
        'Running ' + model_name + ', Add task length = ' + str(length)))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.9,
                                  patience=2,
                                  verbose=1,
                                  mode='auto',
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=1e-6)
    stoponloss = StopOnLoss(monitor='loss', 
                          value=1e-3, 
                          verbose=1)
    checkpoint = MyModelCheckpoint(model_path, monitor='loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='min')

    # ----- Training Model -----
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        callbacks=[reduce_lr, early_stop, print_model_name, checkpoint, stoponloss])
    
    return history


def train_rnn(path, 
              model_name='LSTM', 
              max_len = 50000,
              start_len = 5,
              max_incr=1e100,
              max_entries=3,
              stop_lr=1e-4,
              **kwargs):
    
    logs = {}

    curr_len = start_len
    rnn_model = eval(model_name)
    start_lr = kwargs['learning_rate']
    if not os.path.isfile(model_path):
        hist[rnn_model_name] = add_task(rnn_model, model_name=rnn_model_name, 
                                length=curr_len, load_model=False, 
                                **kwargs)
    iteration = 0
    while curr_len < max_len:
        curr_len = int(curr_len)
        hist[rnn_model_name] = add_task(rnn_model, model_name=rnn_model_name,
                                length=curr_len, load_model=True, 
                                max_entries = max_entries,
                                **kwargs)
        # Save logs
        logs[iteration] = [hist[rnn_model_name].history['loss'], int(curr_len)]
        pickle_out = open(f"{path}{model_name}_H{nb_hid}_logs_s{start_len}_m{max_len}_me{max_entries}","wb")
        pickle.dump(logs, pickle_out)
        iteration = iteration + 1

        if hist[rnn_model_name].history['loss'][-1] < 0.01:
            printer('Error requirements met, increasing length')
            incr_len = min(max_incr,round(curr_len/5))
            incr_len = max(incr_len, 10) # at least increment by 10
            curr_len = curr_len + incr_len
            kwargs['learning_rate'] = start_lr
        else:
            kwargs['learning_rate'] = kwargs['learning_rate']/2
            debug(f'Error requirements not met, maintain length, lower lr to {kwargs["learning_rate"]}')
            if (hist[rnn_model_name].history['loss'][-1] < 0.16) & (kwargs['learning_rate'] < stop_lr):
                printer('Better than chance, but convergence stopped, increasing length')
                incr_len = min(max_incr,round(curr_len/5))
                incr_len = max(incr_len, 10) # at least increment by 10
                curr_len = curr_len + incr_len
                kwargs['learning_rate'] = start_lr
            elif kwargs['learning_rate'] < stop_lr:
                printer('Convergence stopped. Ending')
                return

    if not (curr_len == max_len):
        hist[rnn_model_name] = add_task(rnn_model, model_name=rnn_model_name,
                                length=max_len, load_model=True, 
                                max_entries = max_entries,
                                **kwargs)
    return logs

if __name__ == '__main__':
    # python add_task_curriculum.py --nb_hid=2 --rnn_model=LSTM --start_len=10 --nb_epoch=10 --N_train=10000 --N_test=1 --max_len=10001 --max_entries=10
    # python add_task_curriculum.py --nb_hid=2 --rnn_model=LSTM --start_len=10 --nb_epoch=10 --N_train=10000 --N_test=1 --max_len=11 --max_entries=3
    path = './results/masked_addition/'
    hist = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_hid', type=int, default=32,
                        help='number of hidden layer units')
    parser.add_argument('--nb_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--max_entries', type=int, default=3,
                        help='number of unmasked entries')
    parser.add_argument('--max_len', type=int, default=50000,
                        help='Maximum length of the task')
    parser.add_argument('--start_len', type=int, default=10,
                        help='Starting length of the task')
    parser.add_argument('--rnn_model', type=str, default='lpRNN',
                        help='RNN model type - LSTM, lpRNN, GRU, lpLSTM, IndRNN, lpIndRNN')
    parser.add_argument('--nhybrid', type=int, default=0,
                        help='Number of hybrid units when mixing lpRNN with LSTMs ')
    parser.add_argument('--hybrid_type', type=str, default='LSTM',
                        help='Hybrid type')
    parser.add_argument('--gate_reg', type=str, default=None,
                        help='Gate regularizer')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--start_lr', type=float, default=0.01,
                        help='Start learning rate ')
    parser.add_argument('--stop_lr', type=float, default=0.0001,
                        help='Stop learning rate ')
    parser.add_argument('--nb_epoch', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--max_incr', type=int, default=2000,
                        help='Maximum increment in length')
    parser.add_argument('--N_train', type=int, default=10000,
                        help='Training data size')
    parser.add_argument('--N_test', type=int, default=1000,
                        help='Test data size')
    args = parser.parse_args()

    nb_hid      = args.nb_hid #32
    nb_layers   = args.nb_layers #1
    max_entries = args.max_entries #3 
    max_incr    = args.max_incr #3 
    max_len     = args.max_len #50000
    start_len   = args.start_len #10
    nhybrid     = args.nhybrid #0
    nb_epoch    = args.nb_epoch #10
    batch_size  = args.batch_size #10
    start_lr    = args.start_lr #10
    stop_lr     = args.stop_lr #10
    hybrid_type = args.hybrid_type #LSTM
    activation  = args.activation #relu
    N_test      = args.N_test # test data length
    N_train     = args.N_train # train length
    rnn_model_name = args.rnn_model #'LSTM'
    if args.gate_reg != None:
        gate_regularizer  = eval(args.gate_reg) 
    else:
        gate_regularizer  = None
    # gate_regularizer = l2(0.1)
    kwargs      = {}
    name = rnn_model_name+'_H'+str(nhybrid)
    model_path = os.path.join('./results/masked_addition/',
                              name + '_' + str(nb_hid) + '_' + str(nb_layers) + '.h5')
    kwargs[rnn_model_name] = {
              'learn_retention_ratio': False
              ,'learning_rate':start_lr
              ,'stop_lr':stop_lr
              ,'model_path': model_path
              ,'nb_epoch': nb_epoch
              ,'batch_size': batch_size
              ,'nb_hid': nb_hid
              ,'nb_layers':nb_layers
              ,'hybrid_nunits':nhybrid
              ,'hybrid_type':hybrid_type
              ,'activation':activation
              ,'gate_regularizer':gate_regularizer
              ,'N_train': N_train 
              ,'N_test': N_test
              }
    logs = train_rnn(path=path,
              model_name=rnn_model_name, 
              max_len = max_len,
              start_len = start_len,
              max_incr = max_incr,
              max_entries = max_entries,
              **kwargs[rnn_model_name])