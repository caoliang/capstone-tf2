from tensorflow.keras.layers import Conv2D, Input, MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Permute, Softmax, PReLU
from tensorflow.keras.models import Model
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
    o_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv1')(o_inp)
    o_layer = PReLU(shared_axes=[1, 2], name='prelu1')(o_layer)
    o_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), 
                        padding="same")(o_layer)

    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv2')(o_layer)
    o_layer = PReLU(shared_axes=[1, 2], name='prelu2')(o_layer)
    o_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), 
                        padding="valid")(o_layer)

    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv3')(o_layer)
    o_layer = PReLU(shared_axes=[1, 2], name='prelu3')(o_layer)
    o_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                        padding="same")(o_layer)

    o_layer = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid",
                     name='conv4')(o_layer)
    o_layer = PReLU(shared_axes=[1, 2], name='prelu4')(o_layer)
    
    o_layer = Permute((3,2,1))(o_layer)
    
    o_layer = Flatten()(o_layer)
    o_layer = Dense(256, name='conv5')(o_layer)
    o_layer = PReLU(name='prelu5')(o_layer)

    o_classifier = Dense(2, activation='softmax', name='cls1')(o_layer)
    o_bbox_regress = Dense(4, name='bbox1')(o_layer)
    o_landmark_regress = Dense(10, name='lmk1')(o_layer)

    o_net = Model([o_inp], [o_classifier, o_bbox_regress, o_landmark_regress])
    return o_net


def create_rnet(input_shape=None):
    if input_shape is None:
        input_shape = (24, 24, 3)

    r_inp = Input(input_shape)

    r_layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv1')(r_inp)
    r_layer = PReLU(shared_axes=[1, 2], name='prelu1')(r_layer)
    r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), 
                        padding="same")(r_layer)

    r_layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv2')(r_layer)
    r_layer = PReLU(shared_axes=[1, 2], name='prelu2')(r_layer)
    r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), 
                        padding="valid")(r_layer)

    r_layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid",
                     name='conv3')(r_layer)
    r_layer = PReLU(shared_axes=[1, 2], name='prelu3')(r_layer)
    r_layer = Permute((3,2,1))(r_layer)
    
    r_layer = Flatten()(r_layer)
    r_layer = Dense(128, name='conv4')(r_layer)
    r_layer = PReLU(name='prelu4')(r_layer)

    r_classifier = Dense(2, name='cls1')(r_layer)
    r_classifier = Softmax(axis=1)(r_classifier)

    r_bbox_regress = Dense(4, name='bbox1')(r_layer)

    r_net = Model([r_inp], [r_classifier, r_bbox_regress])

    return r_net


def create_pnet(input_shape=None):
    if input_shape is None:
        input_shape = (None, None, 3)

    p_inp = Input(input_shape)

    p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv1')(p_inp)
    p_layer = PReLU(shared_axes=[1, 2], name='PReLU1')(p_layer)
    p_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                        padding="same")(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv2')(p_layer)
    p_layer = PReLU(shared_axes=[1, 2], name='PReLU2')(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     name='conv3')(p_layer)
    p_layer = PReLU(shared_axes=[1, 2], name='PReLU3')(p_layer)

    p_classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                          activation='softmax', name='cls1')(p_layer)

    p_bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                          name='bbox1')(p_layer)

    p_net = Model([p_inp], [p_classifier, p_bbox_regress])
   
    return p_net


def build_pnet(weights_file = None):
    if weights_file is None:
        weights_file = 'model12.h5'
    
    p_net = create_pnet()
    weights_path = get_weight_path(weights_file)
    p_net.load_weights(weights_path, by_name=True)
    
    return p_net

def build_rnet(weights_file = None):
    if weights_file is None:
        weights_file = 'model24.h5'
    
    r_net = create_pnet()
    weights_path = get_weight_path(weights_file)
    r_net.load_weights(weights_path, by_name=True)
    
    return r_net

def build_onet(weights_file = None):
    if weights_file is None:
        weights_file = 'model48.h5'
    
    o_net = create_pnet()
    weights_path = get_weight_path(weights_file)
    o_net.load_weights(weights_path, by_name=True)
    
    return o_net


def build_mtcnn_nets(weights_files = None):
    if weights_files is None:
        weights_files = ['model12.h5', 'model24.h5', 'model48.h5'] 
                         
    p_net = build_pnet(weights_files[0])
    r_net = build_rnet(weights_files[1])
    o_net = build_onet(weights_files[2])

    return p_net, r_net, o_net