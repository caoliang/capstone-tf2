from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, PReLU
from tensorflow.keras.layers import Flatten, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import _pickle as pickle
import random
import os
import gc

rnet_weights_file = r'..\data\mtcnn_training\model24.h5'

with open(r'..\data\mtcnn_training\24net\cls.imdb','rb') as fid:
    cls = pickle.load(fid)
with open(r'..\data\mtcnn_training\24net\roi.imdb', 'rb') as fid:
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

# input = Input(shape = [12,12,3])
#input = Input(shape = [24,24,3]) # change this shape to [None,None,3] to enable arbitraty shape input

r_inp = Input(shape = [24,24,3])

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

r_layer = Flatten()(r_layer)
r_layer = Dense(128, name='conv4')(r_layer)
r_layer = PReLU(name='prelu4')(r_layer)

classifier = Dense(2, activation='softmax', name='cls1')(r_layer)
classifier = Softmax(axis=1)(classifier)
bbox_regress = Dense(4, name='bbox1')(r_layer)

my_adam = Adam(lr = 0.001)

model = Model([r_inp], [classifier, bbox_regress])
if os.path.exists(rnet_weights_file):
    model.load_weights(rnet_weights_file, by_name=True)
bbox = model.get_layer('bbox1')
bbox_weight = bbox.get_weights()
classifier_dense = model.get_layer('cls1')
cls_weight = classifier_dense.get_weights()

training_parts = 4
training_batches = 80

for i in range(training_parts):
    
    for i_train in range(training_batches):
        
        randx=random.choice([0,1,1])  # still need to run manually on each batch
        # randx = 4
        # randx = random.choice([ 4])
        batch_size = 64
        print ('currently in training macro cycle: ',i_train)
        if 0 == randx:
            model = Model([r_inp], [classifier])
            model.get_layer('cls1').set_weights(cls_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_cls, one_hot_labels, batch_size=batch_size, epochs=1)
            classifier_softmax = model.get_layer('cls1')
            cls_weight = classifier_softmax.get_weights()
    
        if 1 == randx:
            model = Model([r_inp], [bbox_regress])
            model.get_layer('bbox1').set_weights(bbox_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_roi, roi_score, batch_size=batch_size, epochs=1)
            bbox_dense = model.get_layer('bbox1')
            bbox_weight = bbox_dense.get_weights()
    
    gc.collect()

    model = Model([r_inp], [classifier, bbox_regress])
    model.get_layer('bbox1').set_weights(bbox_weight)
    model.get_layer('cls1').set_weights(cls_weight)
    model.save_weights(rnet_weights_file)
            
