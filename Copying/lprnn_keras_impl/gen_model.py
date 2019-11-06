# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda, Flatten
from lprnn_keras_impl import lpRNN, lpLSTM
from keras.regularizers import l1, l2, l1_l2

def lpHybrid(ip, num_lprnn, num_grnn=None, GRNN=LSTM, activation='tanh'):
    if num_grnn == None:
        num_grnn = round(num_lprnn / 10)
        num_lprnn = num_lprnn - num_grnn

    x_rnn = lpRNN(num_lprnn, return_sequences=True, activation=activation)(ip)
    concat = concatenate([ip, x_rnn], axis=-1)
    grnn_op = GRNN(num_grnn, return_sequences=True)(concat)
    x = concatenate([x_rnn, grnn_op], axis=-1)
    return x

def define_model(RNN_model, hybrid_nunits, h_dim, op_dim, num_layers, activation, op_activation,
                  recurrent_initializer, ip, GRNN=LSTM, op_type='seq', learn_retention_ratio=False,
                  gate_regularizer=None, dropout=0):
    # ----- Define Model -----
    h_dim = h_dim - hybrid_nunits
    kwargs = {'dropout':dropout}
    
    if (RNN_model == lpRNN) or (RNN_model == lpLSTM):
        kwargs['learn_retention_ratio'] = learn_retention_ratio


    x = RNN_model(h_dim, return_sequences=True, activation=activation,
                  recurrent_initializer=recurrent_initializer, name=RNN_model.__name__ +'_'+ str(0), **kwargs)(ip)
    if(hybrid_nunits > 0):
        concat = concatenate([x, ip], axis=-1)
        grnn_op = GRNN(hybrid_nunits, return_sequences=True,
                       name=GRNN.__name__ +'_'+ str(1))(concat)
        x = concatenate([x, grnn_op], axis=-1)

    for i in range(1, num_layers - 1):
        if(hybrid_nunits > 0):
            x_rnn = RNN_model(h_dim, return_sequences=True, activation=activation,
                              recurrent_initializer=recurrent_initializer, name=RNN_model.__name__ +'_K'+ str(i+1), **kwargs)(x)
            concat = concatenate([x, x_rnn], axis=-1)
            grnn_op = GRNN(hybrid_nunits, return_sequences=True,
                           name=GRNN.__name__ +'_H'+ str(i))(concat)
            x = concatenate([x_rnn, grnn_op], axis=-1)
        else:
            x = RNN_model(h_dim, return_sequences=True, activation=activation,
                          recurrent_initializer=recurrent_initializer, name=RNN_model.__name__ +'_Y'+ str(i+1), **kwargs)(x)

    if num_layers > 1:
        if(hybrid_nunits > 0):
            x_rnn = RNN_model(h_dim, return_sequences=True, activation=activation,
                              recurrent_initializer=recurrent_initializer, name=RNN_model.__name__ +'_L'+ str(num_layers-1), **kwargs)(x)
            concat = concatenated = concatenate([x, x_rnn], axis=-1)
            grnn_op = GRNN(hybrid_nunits, return_sequences=True,
                           name=GRNN.__name__ +'_'+ str(num_layers-1))(concat)
            x = concatenate([x_rnn, grnn_op], axis=-1)
        else:
            x = RNN_model(h_dim, return_sequences=True, activation=activation,
                          recurrent_initializer=recurrent_initializer, name=RNN_model.__name__ +'_M'+ str(num_layers-1), **kwargs)(x)
    # classification
    if op_type == 'seq':
        x = Dense(op_dim, activation=op_activation,name='dense_'+str(op_dim))(x)
    else:
        x = Lambda(lambda x: x[:, -1:, :])(x)
        x = Flatten()(x)
        x = Dense(op_dim,name='dense_'+str(op_dim), activation=op_activation)(x)
    return x
