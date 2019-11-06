# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import warnings
import numpy as np
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.layers.convolutional_recurrent import ConvRNN2D
from keras import backend as K
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.layers import Recurrent
from keras.utils.generic_utils import get_custom_objects

from .lpRNN_impl import lpRNN
from .LSTM_impl import lpLSTM
