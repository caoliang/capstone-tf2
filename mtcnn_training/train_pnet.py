from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, PReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import _pickle as pickle
import random
import os
import gc


pnet_weights_file = r'..\data\mtcnn_training\model12.h5'
id_train = 0

if id_train == 0:
    cls_imdb_path = r'..\data\mtcnn_training\12net\cls.imdb'
    roi_imdb_path = r'..\data\mtcnn_training\12net\roi.imdb'
else:
    cls_imdb_path = r'..\data\mtcnn_training\12net\cls{0:d}.imdb'.format(
                    id_train)
    roi_imdb_path = r'..\data\mtcnn_training\12net\roi{0:1}.imdb'.format(
                    id_train)

with open(cls_imdb_path, 'rb') as fid:
    cls = pickle.load(fid)
    
with open(roi_imdb_path, 'rb') as fid:
    roi = pickle.load(fid)

ims_cls = []
ims_roi = []
cls_score = []
roi_score = []
for (idx, dataset) in enumerate(cls) :
    ims_cls.append( np.swapaxes(dataset[0],0,2))
    cls_score.append(dataset[1])
for (idx,dataset) in enumerate(roi) :
    ims_roi.append( np.swapaxes(dataset[0],0,2))
    roi_score.append(dataset[2])


ims_cls = np.array(ims_cls)
ims_roi = np.array(ims_roi)
cls_score = np.array(cls_score)
roi_score = np.array(roi_score)


one_hot_labels = to_categorical(cls_score, num_classes=2)

p_inp = Input(shape = [12,12,3])

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

classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                      activation='softmax', name='cls1')(p_layer)
classifier = Reshape((2,))(classifier)   # this layer has to be deleted in order to enalbe arbitraty shape input

bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                      name='bbox1')(p_layer)
bbox_regress = Reshape((4,))(bbox_regress)

my_adam = Adam(lr = 0.00001)

model = Model([p_inp], [classifier, bbox_regress])
if os.path.exists(pnet_weights_file):
    model.load_weights(pnet_weights_file,by_name=True)

bbox = model.get_layer('bbox1')
bbox_weight = bbox.get_weights()
classifier_dense = model.get_layer('cls1')
cls_weight = classifier_dense.get_weights()

training_parts = 32
training_batches = 80

for i in range(training_parts):
    print('******************')
    print('Training parts: ', i)
    print('******************')
    
    for i_train in range(training_batches):
        randx=random.choice([0,1,1])  # still need to run manually on each batch
        # randx = 4
        # randx = random.choice([ 4])
        batch_size = 64
        print ('currently in training macro cycle: ',i_train)
        if 0 == randx:
            model = Model([p_inp], [classifier])
            model.get_layer('cls1').set_weights(cls_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_cls, one_hot_labels, batch_size=batch_size, epochs=1)
            classifier_softmax = model.get_layer('cls1')
            cls_weight = classifier_softmax.get_weights()
    
        if 1 == randx:
            model = Model([p_inp], [bbox_regress])
            model.get_layer('bbox1').set_weights(bbox_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_roi, roi_score, batch_size=batch_size, epochs=1)
            bbox_dense = model.get_layer('bbox1')
            bbox_weight = bbox_dense.get_weights()
    
    gc.collect()
    
    model = Model([p_inp], [classifier, bbox_regress])
    model.get_layer('bbox1').set_weights(bbox_weight)
    model.get_layer('cls1').set_weights(cls_weight)
    model.save_weights(pnet_weights_file)

