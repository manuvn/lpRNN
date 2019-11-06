from keras.models import Model, load_model

from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, CuDNNLSTM, Bidirectional
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import optimizers

from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

from lprnn_keras_impl import lpRNN

def BiRNNSpeechModel(nCategories, samplingrate = 16000, inputLength = 16000, rnn_model=CuDNNLSTM, nunits=64):
    #simple LSTM
    sr = samplingrate
    iLen = inputLength
    
    inputs = Input((iLen,))

    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                             padding='same', sr=sr, n_mels=80,
                             fmin=40.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)

    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs

    x = Permute((2,1,3)) (x)

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    x = Bidirectional(rnn_model(nunits, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(rnn_model(nunits)) (x)

    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)

    output = Dense(nCategories, activation = 'softmax')(x)

    model = Model(inputs=[inputs], outputs=[output])
    
    return model

def RNNSpeechModel(nCategories, samplingrate = 16000, inputLength = 16000, rnn_model=CuDNNLSTM, nunits=64):
    #simple LSTM
    sr = samplingrate
    iLen = inputLength
    
    inputs = Input((iLen,))
 
    x = Reshape((1, -1)) (inputs)
 
    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                             padding='same', sr=sr, n_mels=80,
                             fmin=40.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)
 
    x = Normalization2D(int_axis=0)(x)
 
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
 
    x = Permute((2,1,3)) (x)
 
    x = Conv2D(10, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
 
    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
 
    x = rnn_model(nunits, return_sequences = True) (x) # [b_s, seq_len, vec_dim]
    x = rnn_model(nunits) (x)
 
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
 
    output = Dense(nCategories, activation = 'softmax')(x)
 
    model = Model(inputs=[inputs], outputs=[output])
    
    return model


def SimpleRNNSpeechModel(nCategories, samplingrate = 16000, inputLength = 16000, rnn_model=CuDNNLSTM, nunits=64):
    #simple LSTM
    sr = samplingrate
    iLen = inputLength
    
    inputs = Input((iLen,))
 
    x = Reshape((1, -1)) (inputs)
 
    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                             padding='same', sr=sr, n_mels=80,
                             fmin=40.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)
 
    x = Normalization2D(int_axis=0)(x)
 
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
 
    x = Permute((2,1,3)) (x) 
    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
 
    x = Dense(128, activation='relu') (x) # [b_s, seq_len, vec_dim]
    x = BatchNormalization() (x)
    x = Dense(32, activation='relu') (x) # [b_s, seq_len, vec_dim]
    x = BatchNormalization() (x)
    x = rnn_model(nunits, return_sequences = True) (x) # [b_s, seq_len, vec_dim]
    x = rnn_model(nunits) (x)
 
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
 
    output = Dense(nCategories, activation = 'softmax')(x)
 
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

def lpRNNSpeechModel(nCategories, samplingrate = 16000, inputLength = 16000, nunits=64):
    #simple LSTM
    sr = samplingrate
    iLen = inputLength
    
    inputs = Input((iLen,))
 
    x = Reshape((1, -1)) (inputs)
 
    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                             padding='same', sr=sr, n_mels=80,
                             fmin=40.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)
 
    x = Normalization2D(int_axis=0)(x)
 
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
 
    x = Permute((2,1,3)) (x) 
    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
 
    x = Dense(128, activation='relu') (x) # [b_s, seq_len, vec_dim]
    x = BatchNormalization() (x)
    x = Dense(32, activation='relu') (x) # [b_s, seq_len, vec_dim]
    x = BatchNormalization() (x)
    x = lpRNN(nunits, learn_retention_ratio=False, return_sequences = True) (x) # [b_s, seq_len, vec_dim]
    x = lpRNN(nunits, learn_retention_ratio=True) (x)
 
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
 
    output = Dense(nCategories, activation = 'softmax')(x)
 
    model = Model(inputs=[inputs], outputs=[output])
    
    return model