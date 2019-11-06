# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pickle
import argparse
import sys
import math
# to force CPU - Appears to be much faster than GPU (not using cuDNN)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, CuDNNLSTM
from keras.layers import concatenate
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback, ModelCheckpoint

from lprnn_keras_impl import lpRNN, define_model, lpLSTM
from support import generate_copying_problem
import numpy as np
from support import info, printer, debug

import matplotlib.pyplot as plt

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                 save_best_only=save_best_only, save_weights_only=save_weights_only,
                 mode=mode, period=period)
        
        self.best = 0.04
    def on_epoch_end(self, epoch, logs=None):
        super(MyModelCheckpoint, self).on_epoch_end(epoch=epoch, logs=logs)
        

class StopOnAcc(Callback):
    def __init__(self, monitor='categorical_accuracy', value=0.96, verbose=0):
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

        if current > self.value:
            if self.verbose > 0:
                printer(f"Epoch {epoch}: early stopping accuracy is {current}, Target is {self.value}")
            self.model.stop_training = True
        
        if math.isnan(current):
            if self.verbose > 0:
                printer("NaN accuracy. Killing training.")
            self.model.stop_training = True

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


def data_generator(nsamples=1, Tlen=10, Slen=10, Klen=10, NClasses=11):
    batch_klen = Klen

    while (1):
        batch_slen      = np.random.randint(1,Slen,1)[0] # Pick an slen between 0 and max Slen
        batch_tlen      = 2*(Slen - batch_slen) + Tlen
        sample_weights  = np.ones((nsamples, batch_tlen + 2 * batch_slen))/(batch_slen)
        sample_weights[:, :batch_slen] = 0.0
        sample_weights[:, batch_slen:batch_slen+batch_tlen] = 1/batch_tlen
        # print(f"Batch properties K = {batch_klen}, S = {batch_slen}, T = {batch_tlen}")
        X, Y = generate_copying_problem(
            num_samples=nsamples, Tlen=Tlen, Slen=Slen, Klen=Klen, NClasses=NClasses)
        yield X, Y, sample_weights


def copy_task(RNN_model, model_name='RNN', activation='tanh', nb_layers=1, nb_hid=128,
                nb_epoch=3, batch_size=32, learning_rate=0.1,nbatches=100,
                clipnorm=10, momentum=0, initializer_func=None, hybrid_nunits=0, 
                hybrid_type=LSTM, Tlen=100, Slen=20, Klen=128, NClasses=1024,
                recurrent_initializer='orthogonal', 
                learn_retention_ratio=False, load_model=False,
                model_path=None):
    """Perform LSTM The copying Problem experiment.

    Parameters
    ----------
    RNN_model : handle to the class
    model_name: Just a name

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

    # the model layout picture
    model_pic = os.path.join(model_path, model_name + "-model-pic.png")


    # ----- print mode info -----
#    info("Model Name: ", model_name)
#    info("Number of epochs: ", nb_epoch)
#    info("Batch Size: ", batch_size)
#    info("Number of layers: ", nb_layers)
#    info("Number of hidden units: ", nb_hid)
#    info("Activation: ", activation)
#    info("Recurrent initializer: ", recurrent_initializer)
    
    train_gen = data_generator(
        nsamples=batch_size, Tlen=Tlen, Slen=Slen, Klen=Klen,NClasses=NClasses)
    val_gen = data_generator(nsamples=int(
        batch_size/2), Tlen=Tlen, Slen=Slen, Klen=Klen,NClasses=NClasses)
    X_train, Y_train, _ = next(train_gen)
    X_test, Y_test, _ = next(val_gen)

#    info("Basic dataset statistics")
#    info("X_train shape:", X_train.shape)
#    info("Y_train shape:", Y_train.shape)
#    info("X_test shape:", X_test.shape)
#    info('Y_test shape:', Y_test.shape)

    # setup sequence shape
    input_shape = X_train.shape[1:]

    # ----- Build Model -----
    img_input = Input(shape=input_shape)

    if initializer_func == None:
        initializer_func = keras.initializers.Identity(gain=1.0)

    x = define_model(RNN_model=RNN_model,
                     hybrid_nunits=hybrid_nunits,
                     h_dim=nb_hid,
                     op_dim=NClasses,
                     num_layers=nb_layers,
                     activation=activation,
                     op_activation='softmax',
                     recurrent_initializer=recurrent_initializer,
                     ip=img_input,
                     GRNN=hybrid_type,
                     op_type='seq',
                     learn_retention_ratio=learn_retention_ratio)

    # compile model
    print("[MESSAGE] Compiling model...")
    model = Model(img_input, x)
    model.summary()
    if not os.path.isfile(model_path):
        if load_model == True:
            debug('File does not exist. Creating new.')
    else:
        model.load_weights(model_path, by_name=True)
        
    # ----- Configure Optimizer -----
    # rmsprop = RMSprop(lr=learning_rate, clipnorm=clipnorm)
    opt = SGD(lr=learning_rate, clipnorm=clipnorm, momentum=momentum)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[keras.metrics.categorical_accuracy],
                  sample_weight_mode="temporal")
    print("[MESSAGE] Model is compiled.")

    # Callbacks
    early_stop = EarlyStopping(monitor="loss", 
                               patience=10,
                               verbose=1,
                               min_delta=0.0001,)
    print_model_name = LambdaCallback(on_epoch_begin=lambda batch, 
                                      logs: info('Running ' + 
                                                 model_name + 
                                                 ', Copy task Slen = ' + 
                                                 str(Slen) +
                                                 ', Klen = ' + 
                                                 str(Klen) + 
                                                 ', Tlen = ' + 
                                                 str(Tlen)))
    
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  patience=5,
                                  verbose=1,
                                  mode='auto',
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=1e-7)
    
    checkpoint = MyModelCheckpoint(model_path, 
                                   monitor='loss', 
                                   verbose=1,
                                   save_best_only=True, 
                                   save_weights_only=False, 
                                   mode='min')
    
    stoponacc = StopOnAcc(monitor='categorical_accuracy', 
                          value=0.99, 
                          verbose=1)
    stoponloss = StopOnLoss(monitor='loss', 
                          value=1e-5, 
                          verbose=1)

    # ----- Training Model -----
    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  epochs=nb_epoch,
                                  steps_per_epoch=nbatches,
                                  validation_steps= 1,
                                  callbacks=[reduce_lr, early_stop, 
                                            print_model_name, checkpoint, 
                                            stoponacc, stoponloss])

    # Testing just for visualization
    test_gen = data_generator(nsamples=1, Tlen=Tlen, Slen=Slen, Klen=Klen,NClasses=NClasses)
    X_test, Y_test, _ = next(test_gen)
    pred_ = model.predict(X_test)
    prediction = np.argmax(pred_, axis=-1)
    tgt = np.argmax(Y_test, axis=-1)

    # Plotting the results
    # print('Plotting for case ' + model_name)
    # plt.figure()
    # plt.plot(prediction[0, :], 'r', alpha=0.6, label=model_name + 'Prediction')
    # plt.legend()
    # plt.plot(tgt[0, :], 'k', label='Target')
    # plt.legend()
    # plt.show()
    return history

def train_rnn(  path_root
              , start_tlen
              , start_slen
              , start_klen
              , Tlen_max
              , Slen_max
              , Klen_max
              , model_name='LSTM'
              , **kwargs):
    
    import pickle
    rnn_model = eval(model_name)    
    niters = 0

    curr_klen = start_klen
    curr_slen = start_slen
    curr_tlen = start_tlen

    def incr_tlen(l): 
        incr_l = min(8,l/5) # cap large steps to 10
        incr_l = max(incr_l,1) # step at least 1
        nxt_l = int(incr_l) + l
        print(f'Increment Tlen. old_length is {l}. Next length is {nxt_l}')
        if nxt_l <= Tlen_max:
            return nxt_l
        else:
            return Tlen_max+1
        
    def incr_slen(l):
        incr_l = min(3,l/3) # cap large steps to 3
        incr_l = max(incr_l,1) #avoid fraction
        nxt_l = int(incr_l) + l
        print(f'Increment Slen. old_length is {l}. Next length is {nxt_l}')
        if nxt_l <= Slen_max:
            return nxt_l
        else:
            return Slen_max+1

    def incr_klen(l): 
        nxt_l = round(l + l/2)
        print(f'Increment klen. old_length is {l}. Next length is {nxt_l}')
        if nxt_l <= Klen_max:
            return nxt_l
        else:
            return Klen_max+1

    start_lr = kwargs['learning_rate']

    # not sure stepping through K is a good idea.
    # while curr_slen <= Slen_max:
    iteration = 0
    logs = {}
    curr_klen = start_klen
    while curr_slen <= Slen_max:
        if curr_tlen > Tlen_max:
            curr_tlen = Tlen_max
        while curr_tlen <= Tlen_max:
            info(' Current Klen = '+str(curr_klen)+' Slen = '+str(curr_slen)+' Tlen = '+str(curr_tlen))
            hist[model_name] = copy_task(rnn_model,
                                        model_name=model_name,
                                        Tlen=curr_tlen, 
                                        Slen=curr_slen, 
                                        Klen=curr_klen, 
                                        **kwargs)
            # save logs
            logs[iteration] = [hist[model_name].history['loss'], curr_tlen, curr_slen, curr_klen]
            pickle_out = open(f"{path_root}{model_name}","wb")
            pickle.dump(logs, pickle_out)
            iteration += 1

            if hist[model_name].history['categorical_accuracy'][-1] < 0.96:
                debug('Accuracy not high enough yet.')
                debug('Repeat Klen = '+str(curr_klen)+' Slen = '+str(curr_slen)+' Tlen = '+str(curr_tlen))
                if kwargs['learning_rate'] > 1e-6:
                    kwargs['learning_rate'] = kwargs['learning_rate']/2
                    debug('Lower max learning rate to ',kwargs['learning_rate'])
                else:
                    debug('Convergence failed. Ending')
                    return
                niters = niters + 1    
                debug('Number of iterations = ', niters)    
            else:
                niters = 0
                printer('Done! Save specs and increase complexity.')
                curr_tlen = incr_tlen(curr_tlen)
                kwargs['learning_rate'] = start_lr
                with open(best_specs_path, 'wb') as handle:
                    info('Dumping next iteration specs.')
                    pickle.dump((curr_klen, curr_slen, curr_tlen), 
                                handle, protocol=pickle.HIGHEST_PROTOCOL)
        curr_slen = incr_slen(curr_slen)
        printer('Klen = '+str(curr_klen)+' Slen = '+str(curr_slen)+' Tlen = '+str(curr_tlen))

        if (curr_slen == Slen_max) and (curr_klen == Klen_max) and (curr_tlen == Tlen_max):
            printer('FINISHED!!!!')
            printer('Klen = '+str(curr_klen)+' Slen = '+str(curr_slen)+' Tlen = '+str(curr_tlen))
            return


if __name__ == '__main__':
    # python copy_task_curriculum.py  --nb_hid=256 --nb_layers=1 --rnn_model=lpRNN --clipnorm=1000
    # python copy_task_curriculum.py  --nb_hid=128 --nb_layers=1 --rnn_model=LSTM --clipnorm=1 --lr=0.005
    # For LSTM set stop at -.99. lpRNN cannot coverge to such a low error.
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    path = './results/copying/'
    hist = {}

    parser = argparse.ArgumentParser()

    # NETWORK params
    parser.add_argument('--nb_hid', type=int, default=128,
                        help='number of hidden layer units')
    parser.add_argument('--nb_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--nb_hybrid', type=int, default=0,
                        help='number of hybrid nodes per layer')
    parser.add_argument('--hybrid_type', type=str, default='LSTM',
                        help='Type of hybrid node')
    parser.add_argument('--rnn_model', type=str, default='LSTM',
                        help='RNN model type - LSTM, lpRNN, GRU, lpLSTM, IndRNN, lpIndRNN')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')

    # Optimizer params
    parser.add_argument('--lr', type=float, default=0.01, #0.01 seems to cause nan loss results
                        help='Learning rate')
    parser.add_argument('--nb_epoch', type=int, default=1000,
                        help='Number of epochs. We stop when accuracy has converged..')
    parser.add_argument('--nbatches', type=int, default=1000,
                        help='Number of batches per epoch. Keep batch size small and \
                        nbatches high to help better curriculum training.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. Keep batch size small and \
                        nbatches high to help better curriculum training.')
    parser.add_argument('--clipnorm', type=float, default=10,
                        help='Clipping factor for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='MOmentum factor for optimizer')

    # Pre-training 
    parser.add_argument('--load_specs', type=str, default='True',
                        help='Load specs of best (S, K, T) from a previous run')
    parser.add_argument('--load_model', type=bool, default=True,
                        help='Load weights from previous run')

    # Task params
    parser.add_argument('--K', type=int, default=8,
                        help='K = alphabet size')
    parser.add_argument('--S', type=int, default=8,
                        help='S = Length of string to be copied')
    parser.add_argument('--T', type=int, default=200,
                        help='T = Length of the memory task')
    parser.add_argument('--Kstart', type=int, default=8,
                        help='Kstart = alphabet size to use at start of sim')
    parser.add_argument('--Sstart', type=int, default=8,
                        help='Sstart = Length of string to be copied to use at start of sim')
    parser.add_argument('--Tstart', type=int, default=3,
                        help='Tstart = Length of the memory task to use at start of sim')
    parser.add_argument('--nclasses', type=int, default=9,
                        help='Number of classes allowed. K to be less than nclasses.')

    args = parser.parse_args()

    if args.nclasses < args.K+1:
          sys.exit(f'Nclasses = {args.nclasses} should be 1 more than K = {args.K}')

    nb_hid = args.nb_hid
    nb_layers = args.nb_layers
    kwargs = {}
    rnn_model_name = args.rnn_model
    nhybrid = args.nb_hybrid

    Kmax = args.K
    Smax = args.S
    Tmax = args.T

    load_specs = eval(args.load_specs)

    name = rnn_model_name+'_H'+str(nb_hid)+'_HL'+str(nb_layers)   
    path_root =  os.path.join('./results/copying/', name)
    best_specs_path = path_root + '_res'+'.pk'
    model_path = os.path.join(path_root + '.h5')    

    kwargs[rnn_model_name] = {
              'learn_retention_ratio': False
              ,'model_path': model_path
              ,'nb_hid': nb_hid
              ,'nb_layers':nb_layers
              ,'hybrid_nunits':nhybrid
              ,'hybrid_type':args.hybrid_type
              ,'load_model': args.load_model
              ,'learning_rate' : args.lr
              ,'clipnorm' : args.clipnorm
              ,'momentum' : args.momentum              
              ,'nb_epoch': args.nb_epoch
              ,'batch_size' : args.batch_size
              ,'nbatches' : args.nbatches
              ,'NClasses' : args.nclasses
              ,'activation':'relu'
              }
    print("Number of epochs is ", kwargs[rnn_model_name]['nb_epoch'])
      
    # Load best specs on record.    
    if load_specs and os.path.isfile(best_specs_path):
        info('Loading previous run specs')
        with open(best_specs_path, 'rb') as handle:
            start_klen, start_slen, start_tlen = pickle.load(handle)
        info('Start Klen = ' + str(start_klen) + ', Slen = ' + str(start_slen) + ',Tlen = '+str(start_tlen))
    else:
        start_tlen = args.Tstart
        start_slen = args.Sstart
        start_klen = args.Kstart

    if start_klen > Kmax:
        start_klen = Kmax
    if start_tlen > Tmax:
        start_tlen = Tmax
    if start_slen > Smax:
        start_slen = Smax
        

    train_rnn(model_name=rnn_model_name
              , path_root=path_root
              , Tlen_max=Tmax
              , Slen_max=Smax
              , Klen_max=Kmax
              , start_tlen = start_tlen
              , start_slen = start_slen
              , start_klen = start_klen
              , **kwargs[rnn_model_name])