import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D
from tensorflow.keras.layers import Reshape, Activation, Flatten
from tensorflow.keras.layers import Dense, Permute, Softmax, PReLU
from tensorflow.keras.models import Model, Sequential
import numpy as np
import os

def get_weight_path(weight_filename):
    source_dir = os.path.dirname(os.path.abspath(__file__))
    #print(source_dir)
    weight_path = os.path.join(source_dir, weight_filename)
    print(weight_path)
    return weight_path

def create_onet(input_shape=None):
    if input_shape is None:
        input_shape = (48, 48, 3)

    o_inp = Input(input_shape)
    o_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_inp)
    o_layer = PReLU(shared_axes=[1, 2])(o_layer)
    o_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(o_layer)

    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_layer)
    o_layer = PReLU(shared_axes=[1, 2])(o_layer)
    o_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(o_layer)

    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_layer)
    o_layer = PReLU(shared_axes=[1, 2])(o_layer)
    o_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(o_layer)

    o_layer = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid")(o_layer)
    o_layer = PReLU(shared_axes=[1, 2])(o_layer)

    o_layer = Flatten()(o_layer)
    o_layer = Dense(256)(o_layer)
    o_layer = PReLU()(o_layer)

    o_layer_out1 = Dense(2)(o_layer)
    o_layer_out1 = Softmax(axis=1)(o_layer_out1)
    o_layer_out2 = Dense(4)(o_layer)
    o_layer_out3 = Dense(10)(o_layer)

    o_net = Model(o_inp, [o_layer_out2, o_layer_out3, o_layer_out1])
    return o_net


def create_rnet (input_shape=None):
    if input_shape is None:
        input_shape = (24, 24, 3)

    r_inp = Input(input_shape)

    r_layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid")(r_inp)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(r_layer)

    r_layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid")(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(r_layer)

    r_layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid")(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    r_layer = Flatten()(r_layer)
    r_layer = Dense(128)(r_layer)
    r_layer = PReLU()(r_layer)

    r_layer_out1 = Dense(2)(r_layer)
    r_layer_out1 = Softmax(axis=1)(r_layer_out1)

    r_layer_out2 = Dense(4)(r_layer)

    r_net = Model(r_inp, [r_layer_out2, r_layer_out1])

    return r_net


def create_pnet(input_shape=None):
    if input_shape is None:
        input_shape = (None, None, 3)

    p_inp = Input(input_shape)

    p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_inp)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)
    p_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(p_layer)
    p_layer_out1 = Softmax(axis=3)(p_layer_out1)

    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1))(p_layer)

    p_net = Model(p_inp, [p_layer_out2, p_layer_out1])
   
    return p_net


def build_mtcnn_nets(weights_files = None):
    if weights_files is None:
        weights_files = ['pnet.h5', 'rnet.h5', 'onet.h5'] 
                         
    p_net = create_pnet()
    r_net = create_rnet()
    o_net = create_onet()

    nets = [p_net, r_net, o_net]
    
    for index in range(len(nets)):
        weights_path = get_weight_path(weights_files[index])
        nets[index].load_weights(weights_path)

    return p_net, r_net, o_net