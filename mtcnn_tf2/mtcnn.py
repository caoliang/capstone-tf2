"""Main script. Contain model definition and training code."""
import os
import numpy as np

import tensorflow as tf
from tensorflow.compat.v1 import parse_single_example, FixedLenFeature
from tensorflow.compat.v1 import decode_raw

from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Lambda
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Dense, Permute, Softmax, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class NetWork(object):

    def __init__(self, mode='data', rnd_thresholds=None, weights_path=None):

        self.mode = mode
        self.rnd_thresholds = rnd_thresholds
        self.weights_path = weights_path
        
        # Network model
        self.net_model = self.setup()
        # Load weights if available
        self.load()

    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self):
        if os.path.exists(self.weights_path):
            self.net_model.load_weights_file(self.weights_path, by_name=True)

    def get_net_model(self):
        return self.net_model

    def get_data_input(self, net_inputs, thresholds=None):
        random_int = tf.random_uniform([1])
        net_inputs_len = len(net_inputs)
        if net_inputs_len == 2:
            if thresholds is None: thresholds = [0.5]
            
            condition = random_int[0] > tf.constant(thresholds[0])
            val = tf.case({condition: lambda: net_inputs[0]},
                          default=lambda: net_inputs[1])
        elif net_inputs_len == 3:
            if thresholds is None: thresholds = [0.4, 0.9]
            
            condition0 = random_int[0] > tf.constant(thresholds[0])
            condition1 = random_int[0] > tf.constant(thresholds[1])
            
            val = tf.case({condition1: lambda: net_inputs[1],
                           condition0: lambda: net_inputs[2],
                           },
                           default=lambda: net_inputs[2],
                           exclusive = False)
        else:
            raise Exception(f'Invalid net_inputs size: {len(net_inputs)}')
        
        val.set_shape(net_inputs[0].shape)
        return [val,random_int]# tuple (output,random_int ) is NOT allowed


class PNet(NetWork):

    def create_pnet(self):
        if self.mode == 'data':
            p_inp = Input(shape = [12, 12, 3])
        else:
            train_inputs = [Input(shape = [12, 12, 3]),
                            Input(shape = [12, 12, 3])]
            p_inp = Lambda(self.get_data_input)(train_inputs)
    
        p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv1')(p_inp)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU1')(p_layer)
        p_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                            padding="same")(p_layer)
    
        p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv2')(p_layer)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU2')(p_layer)
    
        p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv3')(p_layer)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU3')(p_layer)
    
        if self.mode == 'data':
            p_classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                                  activation='softmax', name='cls1')(p_layer)
            p_classifier = Reshape((2,))(p_classifier)
                
            p_bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                                  name='bbox1')(p_layer)
            p_bbox_regress = Reshape((4,))(p_bbox_regress)
        
            p_net = Model([p_inp], [p_classifier, p_bbox_regress])
        else:
            p_classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                              activation='softmax', name='cls1')(p_layer)
            p_classifier = Reshape((2,))(p_classifier)
            p_bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                              name='bbox1')(p_layer)
            p_bbox_regress = Reshape((4,))(p_bbox_regress)
            p_net = Model([p_inp], [p_classifier, p_bbox_regress])
       
        return p_net
    
    def setup(self):
        return self.create_pnet()


def read_and_decode(serialized_data):
    context, sequence = parse_single_example(
        serialized_data,
        features={
            'image_raw': FixedLenFeature([], tf.string),
            'label_raw': FixedLenFeature([], tf.string),
        })
    print('context: ')
    print(context)
    image = decode_raw(context[0], tf.uint8)
    print('image: ' + str(image))
    image = tf.cast(image, tf.float32)
    print('cast: ' + str(image))
    label = decode_raw(context[1], tf.float32)
    print('cast: ' + str(label))

    return [image, label]

#    image = (image - 127.5) * (1. / 128.0)
#    image.set_shape([shape * shape * 3])
#    image = tf.reshape(image, [shape, shape, 3])
#    
#    if label_type == 'cls':
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_flip_up_down(image)
#        label.set_shape([2])
#    elif label_type == 'bbx':
#        label.set_shape([4])
#    elif label_type == 'pts':
#        label.set_shape([10])
#
#    return [image, label]


def prepare_train_inputs(tfrecords_filename, batch_size, 
                         num_epochs, label_type, shape):
    #capacity=1000 + 3 * batch_size
    
    images, labels = tf.data.TFRecordDataset(tfrecords_filename) \
                    .map(read_and_decode)
                    #.shuffle(capacity) \
                    #.batch(batch_size) \
                    #.repeat(num_epochs) \
                    
    return images, labels

def mtcnn_loss(net_type, train_tasks=2, rand_threshold=[0.5]): # need to make sure input type
    random_int = tf.random_uniform([1])
    
    print_progress = False
    
    if print_progress: 
        random_int = tf.Print(random_int[0], ['random in cls',random_int])
    if train_tasks == 2:
        condition0 = random_int[0] > tf.constant(rand_threshold[0])
        condition1 = not condition0
    else:
        condition0 = random_int[0] > tf.constant(rand_threshold[0])
        condition1 = random_int[0] > tf.constant(rand_threshold[1])
    
    if net_type == 'cls':
        def lossfun(y_true, y_pred):

            y_mean_square = backend.mean(backend.square(y_true), axis=-1)
    
            if train_tasks == 2:
                if print_progress: 
                    tf.Print( condition0, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition0:', condition0 ])
                val= tf.case(
                        { 
                            condition0: lambda: backend.mean(
                                backend.square(y_pred - y_true), 
                                axis=-1) 
                        },
                        default = lambda: 0 * y_mean_square,
                        exclusive=False )

                if print_progress: 
                    tf.Print( val, [ 'cls loss out:',val ])
            else:
                if print_progress: 
                    tf.Print( condition0, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition0:', condition0 ])
                    tf.Print( condition1, [ 
                            'rand int[1]:', random_int[1],
                            ' tf.constant:', tf.constant(rand_threshold[1]),
                            ' condition1:', condition1 ])
                val= tf.case(
                        { 
                            condition1: lambda: backend.mean(
                                    backend.square(y_pred - y_true), 
                                    axis=-1),
                            condition0: lambda: 0 * y_mean_square
                        },
                        default = lambda: 0 * y_mean_square,
                        exclusive=False )
                if print_progress: 
                    tf.Print( val, [ 'cls loss out:',val ]) 
               
            val.set_shape(y_mean_square.shape)
            
            return val
        
    elif net_type == 'bbx':
        def lossfun(y_true, y_pred):
            
            y_mean_square = backend.mean(backend.square(y_true), axis=-1)
            
            if train_tasks == 2:
                
                if print_progress: 
                    tf.Print( condition1, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition1:', condition1 ])
                val= tf.case(
                        { 
                            condition1: lambda: backend.mean(
                                backend.square(y_pred - y_true), 
                                axis=-1) 
                        },
                        default = lambda: 0 * y_mean_square,
                        exclusive=False )

                if print_progress: 
                    tf.Print( val, [ 'bbx loss out:',val ])
            else:
                if print_progress: 
                    tf.Print( condition0, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition0:', condition0 ])
                    tf.Print( condition1, [ 
                            'rand int[1]:', random_int[1],
                            ' tf.constant:', tf.constant(rand_threshold[1]),
                            ' condition1:', condition1 ])
                val= tf.case(
                        { 
                            condition1: lambda: 0 * y_mean_square,
                            condition0: lambda: lambda: backend.mean(
                                    backend.square(y_pred - y_true), 
                                    axis=-1),
                        },
                        default = lambda: 0 * y_mean_square,
                        exclusive=False )
                if print_progress: 
                    tf.Print( val, [ 'bbx loss out:', val ]) 
                    
            val.set_shape(y_mean_square.shape)
    
            return val
    else :
        def lossfun(y_true, y_pred):
            
            y_mean_square = backend.mean(backend.square(y_true), axis=-1)
 
            if print_progress: 
                tf.Print( condition0, [ 
                        'rand int[0]:', random_int[0],
                        ' tf.constant:', tf.constant(rand_threshold[0]),
                        ' condition0:', condition0 ])
                tf.Print( condition1, [ 
                        'rand int[1]:', random_int[1],
                        ' tf.constant:', tf.constant(rand_threshold[1]),
                        ' condition1:', condition1 ])
            val= tf.case(
                    { 
                        condition1: lambda: 0 * y_mean_square,
                        condition0: lambda: 0 * y_mean_square,
                    },
                    default = lambda: backend.mean(
                                backend.square(y_pred - y_true), 
                                axis=-1),
                    exclusive=False )
                    
            if print_progress: 
                tf.Print( val, [ 'pts loss out:',val])                
            
            val.set_shape(y_mean_square.shape)
            
            return val

    return lossfun


def accuracy_mean(y_pred,y_true):
    return backend.mean(y_true)


def train_net(Net, training_data_files, base_lr,
              num_epochs=1, batch_size=64, 
              save_filename=None):

    rnd_seed = 49
    np.random.seed(rnd_seed)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    tasks = ['cls', 'bbx', 'pts']
    if Net.__name__ == 'PNet':
        shape_size = 12
        train_mode = 2
    elif Net.__name__ == 'RNet':
        shape_size = 24
        train_mode = 2
    elif Net.__name__ == 'ONet':
        shape_size = 48
        train_mode = 3
    else:
        raise Exception('Invalid training net model')
    
    for index in range(train_mode):
        data_images, data_labels = prepare_train_inputs(
            tfrecords_filename=training_data_files[index],
            batch_size=batch_size,
            num_epochs=num_epochs,
            label_type=tasks[index],
            shape=shape_size)
        
        X_train, X_test, y_train, y_test = train_test_split(
                data_images, data_labels,
                test_size=1/3,
                random_state=rnd_seed)
        
        train_images.append(X_train)
        test_images.append(X_test)
        
        if tasks[index] == 'cls':
            y_train = to_categorical(y_train, num_classes=2)
            y_test = to_categorical(y_test, num_classes=2)
                
        train_labels.append(y_train)
        test_labels.append(y_test)
        
    mtcnn_adam = Adam(lr = base_lr)

    mtcnn_net = Net(mode='train', weights_path=save_filename)
    mtcnn_net.compile(loss=[mtcnn_loss('cls'), mtcnn_loss('bbx')],
                      optimizer=mtcnn_adam, 
                      metrics=[accuracy_mean, accuracy_mean])    
    
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(  filepath=save_filename,
                                    save_weights_only=True, 
                                    verbose=1,
                                    save_best_only=True,
                                    mode='max')

    # Log the epoch detail into csv
    csv_logger = CSVLogger(save_filename + '.csv')
    
    mtcnn_net.fit(train_images, train_labels, 
                  validation_data=(test_images, test_labels),
                  batch_size=batch_size, 
                  epochs=num_epochs, 
                  callbacks=[cp_callback, csv_logger])

def train_pnet(training_data_files, base_lr,
               num_epochs, save_filename=None):

    train_net(Net=PNet,
              training_data_files=training_data_files,
              base_lr=base_lr,
              num_epochs=num_epochs,
              save_filename=save_filename)


if __name__ == '__main__':

    model_filename = '../data/mtcnn_training/saved_model/pnet.hdf5'
    training_data_files = ['../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_cls.tfrecords',
                     '../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_bbx.tfrecords']
    train_pnet(training_data_files=training_data_files,
               base_lr=0.0001,
               num_epochs=1,
               save_filename=model_filename)


