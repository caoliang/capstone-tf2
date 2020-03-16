"""Main script. Contain model definition and training code."""
import os
import numpy as np
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Lambda
from tensorflow.keras.layers import Reshape, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical, plot_model



class NetWork(object):

    def __init__(self, mode='data',  
                 weights_path=None,
                 train_inputs=None,
                 net_thresholds=None):

        self.mode = mode
        self.weights_path = weights_path
        self.train_inputs = train_inputs
        self.net_thresholds = net_thresholds
        
        # Network model
        self.net_model = self.setup()
        # Load weights if available
        self.load()

    def setup(self, train_inputs=None):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self):
        if os.path.exists(self.weights_path):
            self.net_model.load_weights(self.weights_path, by_name=True)

    def get_net_model(self):
        return self.net_model

    def get_train_input(self, net_inputs):
        thresholds = self.net_thresholds
        
        random_int = tf.random.uniform([1])
        net_inputs_len = len(net_inputs)
        if net_inputs_len == 2:
            if thresholds is None: 
                thresholds = [0.5]
            
            condition0 = random_int[0] > tf.constant(thresholds[0])
            
            val = tf.case([(condition0, (lambda: net_inputs[0]))],
                           default=(lambda: net_inputs[1]),
                           exclusive = False)
            
        elif net_inputs_len == 3:
            if thresholds is None: 
                thresholds = [0.4, 0.9]
            
            condition0 = random_int[0] > tf.constant(thresholds[0])
            condition1 = random_int[0] > tf.constant(thresholds[1])
            
            val = tf.case([(condition1, lambda: net_inputs[1]),
                           (condition0, lambda: net_inputs[2])],
                           default=lambda: net_inputs[2],
                           exclusive = False)
        else:
            raise Exception(f'Invalid net_inputs size: {len(net_inputs)}')
        
        val.set_shape(net_inputs[0].shape)
        return [val,random_int]


class PNet(NetWork):

    def create_pnet(self):
        if self.mode == 'data':
            p_inp = Input(shape = [12, 12, 3])
        else:
            train_inp = [Input(shape = [12, 12, 3]),
                         Input(shape = [12, 12, 3])]
            (p_inp, random_int) = Lambda(self.get_train_input)(train_inp)
    
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
            p_net = Model(train_inp, [p_classifier, p_bbox_regress])
       
        return p_net
    
    def setup(self):
        return self.create_pnet()


def read_and_decode(serialized_data, label_type, shape):
    print('serialized_data: ' + serialized_data)
    print(f'label_type: {label_type}, shape: {shape}')
    features = {
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),}
    parsed_dataset = tf.io.parse_single_example(
            serialized_data, features)

    print(f'parsed_dataset: {parsed_dataset}')
    image = tf.io.decode_raw(parsed_dataset['image_raw'], tf.uint8)
    #print(f'image: {image}')
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) * (1. / 128.0)
    image = tf.reshape(image, [shape, shape, 3])
    
    label_shape = get_label_shape(label_type)    
    label = tf.io.decode_raw(parsed_dataset['label_raw'], tf.float32)
    label.set_shape([label_shape])
        
    if label_type == 'cls':
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    
    return image, label

def get_label_shape(label_type):
    label_shape = -1
    if label_type == 'cls':
        label_shape = 2
    elif label_type == 'bbx':
        label_shape = 4
    elif label_type == 'pts':
        label_shape = 10
    else:
        raise Exception(f"Invalid label type: {label_type}")
        
    return label_shape

# max_data_size 
# - 0 or negative number -> no limit
# - Positive number -> limit data length
def prepare_train_test_inputs(tfrecords_filename, batch_size, 
                         num_epochs, label_type, shape,
                         max_data_size=10):
    dataset = tf.data.TFRecordDataset(tfrecords_filename)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(batch_size * 4)
    
    map_func = lambda x: read_and_decode(x, label_type, shape)
    parsed_dataset = dataset.map(map_func)
    parsed_dataset = parsed_dataset.batch(batch_size, drop_remainder=True)
    
    train_images_dataset = []
    train_labels_dataset = []
    test_images_dataset = []
    test_labels_dataset = []
    
    batch_count = 0
    
    for batch_line in parsed_dataset:
        batch_count += 1
        images, labels = batch_line
        
        if max_data_size > 0 and batch_count * batch_size > max_data_size:
            break
        
        # Test : Training = 1 : 3
        if batch_count % 4 == 0:
            test_images_dataset.extend(images)
            test_labels_dataset.extend(labels)
        else:
            train_images_dataset.extend(images)
            train_labels_dataset.extend(labels)
        
    train_images_dataset = np.array(train_images_dataset)
    print(f"train_images_dataset shape: {train_images_dataset.shape}")

    test_images_dataset = np.array(test_images_dataset)
    print(f"test_images_dataset shape: {test_images_dataset.shape}")

#    label_shape = get_label_shape(label_type)
    train_labels_dataset = np.array(train_labels_dataset)
    test_labels_dataset = np.array(test_labels_dataset)
    
#    if label_type == 'cls':
#        train_labels_dataset = to_categorical(train_labels_dataset, 
#                                              num_classes=2)
#        test_labels_dataset = to_categorical(test_labels_dataset, 
#                                              num_classes=2)
        
    print(f"train_labels_dataset shape: {train_labels_dataset.shape}")
    print(f"test_labels_dataset shape: {test_labels_dataset.shape}")

    return train_images_dataset, test_images_dataset, \
           train_labels_dataset, test_labels_dataset


def mtcnn_loss(net_type, train_tasks=2, rand_threshold=[0.5]): # need to make sure input type
    random_int = tf.random.uniform([1])
    
    print_progress = False
    
    if print_progress: 
        random_int = tf.Print(random_int[0], ['random in cls',random_int])
    if train_tasks == 2:
        condition0 = random_int[0] > tf.constant(rand_threshold[0])
        condition1 = random_int[0] <= tf.constant(rand_threshold[0])
    else:
        condition0 = random_int[0] > tf.constant(rand_threshold[0])
        condition1 = random_int[0] > tf.constant(rand_threshold[1])
    
    if net_type == 'cls':
        def lossfun(y_true, y_pred):

            y_mean_square = backend.mean(backend.square(y_true), axis=-1)

            cls_func = lambda: backend.mean(
                                      backend.square(y_pred - y_true), 
                                      axis=-1) 
            cls_zero = lambda: 0 * y_mean_square
    
            if train_tasks == 2:
                if print_progress: 
                    tf.Print( condition0, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition0:', condition0 ])

                val= tf.case([(condition0, cls_func)],
                              default = cls_zero,
                              exclusive=False)

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
                val= tf.case([(condition1, cls_func),
                              (condition0, cls_zero)],
                             default = cls_zero,
                             exclusive=False)
                if print_progress: 
                    tf.Print( val, [ 'cls loss out:',val ]) 
               
            val.set_shape(y_mean_square.shape)
            
            return val
        
    elif net_type == 'bbx':
        def lossfun(y_true, y_pred):
            
            y_mean_square = backend.mean(backend.square(y_true), axis=-1)
            
            bbx_func = lambda: backend.mean(
                          backend.square(y_pred - y_true), 
                          axis=-1) 
            bbx_zero = lambda: 0 * y_mean_square
            
            if train_tasks == 2:
                
                if print_progress: 
                    tf.Print( condition1, [ 
                            'rand int[0]:', random_int[0],
                            ' tf.constant:', tf.constant(rand_threshold[0]),
                            ' condition1:', condition1 ])
                val= tf.case([(condition1, bbx_func)],
                              default = bbx_zero,
                              exclusive=False)

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
                val= tf.case([(condition1, bbx_zero),
                              (condition0, bbx_func)],
                              default = bbx_zero,
                              exclusive=False)
                if print_progress: 
                    tf.Print( val, [ 'bbx loss out:', val ]) 
                    
            val.set_shape(y_mean_square.shape)
    
            return val
    else :
        def lossfun(y_true, y_pred):
            
            y_mean_square = backend.mean(backend.square(y_true), axis=-1)
 
            pts_func = lambda: backend.mean(
                      backend.square(y_pred - y_true), 
                      axis=-1) 
            pts_zero = lambda: 0 * y_mean_square
    
            if print_progress: 
                tf.Print( condition0, [ 
                        'rand int[0]:', random_int[0],
                        ' tf.constant:', tf.constant(rand_threshold[0]),
                        ' condition0:', condition0 ])
                tf.Print( condition1, [ 
                        'rand int[1]:', random_int[1],
                        ' tf.constant:', tf.constant(rand_threshold[1]),
                        ' condition1:', condition1 ])
            val= tf.case([(condition1, pts_zero),
                          (condition0, pts_zero)],
                          default = pts_func,
                          exclusive=False)
                    
            if print_progress: 
                tf.Print( val, [ 'pts loss out:',val])                
            
            val.set_shape(y_mean_square.shape)
            
            return val

    return lossfun


def accuracy_mean(y_pred,y_true):
    return backend.mean(y_true)


def train_net(Net, training_data_files, base_lr,
              num_epochs=1, batch_size=64, 
              save_filename=None,
              max_data_size=1280):

    rnd_seed = 49
    np.random.seed(rnd_seed)
    tf.random.set_seed(rnd_seed)
    
    print(f"max_data_size: {max_data_size}")

    base_dir_name = os.path.dirname(save_filename)
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
    print(f"base_dir_name: {base_dir_name}")
    
    base_model_name = os.path.splitext(os.path.basename(save_filename))[0]
    print(f"base_model_name: {base_model_name}")
    
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
        X_train, X_test, y_train, y_test = prepare_train_test_inputs(
            tfrecords_filename=training_data_files[index],
            batch_size=batch_size,
            num_epochs=num_epochs,
            label_type=tasks[index],
            shape=shape_size,
            max_data_size=max_data_size)
        
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        train_images.append(X_train)
        test_images.append(X_test)
                
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        train_labels.append(y_train)
        test_labels.append(y_test)

        
    mtcnn_adam = Adam(lr = base_lr)

    mtcnn_net = Net(mode='train', weights_path=save_filename)
    mtcnn_net_model = mtcnn_net.get_net_model()
    mtcnn_net_model.compile(loss=[mtcnn_loss('cls'), mtcnn_loss('bbx')],
                            optimizer=mtcnn_adam, 
                            metrics=[accuracy_mean, accuracy_mean])    
    
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(  filepath=save_filename,
                                    save_weights_only=True, 
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min')

    # Log the epoch detail into csv
    csv_out_path = os.path.join(base_dir_name, base_model_name + '.csv')
    csv_logger = CSVLogger(csv_out_path)
    
    mtcnn_net_model.fit(train_images, train_labels, 
                        validation_data=(test_images, test_labels),
                        batch_size=batch_size, 
                        epochs=num_epochs, 
                        callbacks=[cp_callback, csv_logger])

    plot_training_model(csv_out_path)
    
    # Save model in PDF file
    model_pdf_file = os.path.join(base_dir_name, base_model_name + '.pdf')
    save_training_model(mtcnn_net_model, model_pdf_file)


def plot_training_model(training_csv_path):
    records = pd.read_csv(training_csv_path)
    plt.figure()
    plt.plot(records['val_loss'])
    plt.plot(records['loss'])
    plt.yticks([0,0.20,0.40,0.60,0.80,1.00])
    plt.title('Loss value',fontsize=12)
    
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.show()


def save_training_model(model, model_pdf_file):
    try:
        plot_model(model, 
           to_file=model_pdf_file, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')
    except Exception as exp:
        #print(f"Save model pdf error: {exp}")
        pass


def train_pnet(training_data_files, base_lr,
               num_epochs, 
               batch_size,
               save_filename=None,
               max_data_size=1280):

    train_net(Net=PNet,
              training_data_files=training_data_files,
              base_lr=base_lr,
              num_epochs=num_epochs,
              batch_size=batch_size,
              save_filename=save_filename,
              max_data_size=max_data_size)


if __name__ == '__main__':

    model_filename = '../data/mtcnn_training/saved_model/pnet.hdf5'
    training_data_files = ['../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_cls.tfrecords',
                     '../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_bbx.tfrecords']
    
    # max_data_size
    # 0 or Negative - All data were used
    # Positive - Limit number of data were used
    train_pnet(training_data_files=training_data_files,
               base_lr=0.0001,
               num_epochs=3,
               batch_size=64,
               save_filename=model_filename,
               max_data_size=300)


