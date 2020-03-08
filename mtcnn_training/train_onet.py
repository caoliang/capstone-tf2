from tensorflow.keras.layers import Conv2D, Input, MaxPool2D 
from tensorflow.keras.layers import Flatten, Dense, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import _pickle as pickle
import random
import os
import gc


onet_weights_file = r'..\data\mtcnn_training\model48.h5'

id_train = 1
with open(r'..\data\mtcnn_training\48net\cls{0:d}.imdb'.format(id_train),'rb') as fid:
    cls = pickle.load(fid)
with open(r'..\data\mtcnn_training\48net\pts{0:d}.imdb'.format(id_train),'rb') as fid:
     pts = pickle.load(fid)
with open(r'..\data\mtcnn_training\48net\roi{0:d}.imdb'.format(id_train), 'rb') as fid:
    roi = pickle.load(fid)
ims_cls = []
ims_pts = []
ims_roi = []
cls_score = []
pts_score = []
roi_score = []
for (idx, dataset) in enumerate(cls) :
    ims_cls.append( np.swapaxes(dataset[0],0,2))
    cls_score.append(dataset[1])
for (idx,dataset) in enumerate(roi) :
    ims_roi.append( np.swapaxes(dataset[0],0,2))
    roi_score.append(dataset[2])
for (idx,dataset) in enumerate(pts) :
    ims_pts.append( np.swapaxes(dataset[0],0,2))
    pts_score.append(dataset[3])

ims_cls = np.array(ims_cls)
ims_pts = np.array(ims_pts)
ims_roi = np.array(ims_roi)
cls_score = np.array(cls_score)
pts_score = np.array(pts_score)
roi_score = np.array(roi_score)


one_hot_labels = to_categorical(cls_score, num_classes=2)

o_inp = Input(shape = [48,48,3])
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

o_layer = Flatten()(o_layer)
o_layer = Dense(256, name='conv5')(o_layer)
o_layer = PReLU(name='prelu5')(o_layer)

classifier = Dense(2, activation='softmax', name='cls1')(o_layer)
bbox_regress = Dense(4, name='bbox1')(o_layer)
landmark_regress = Dense(10, name='lmk1')(o_layer)

my_adam = Adam(lr = 0.00003)


model = Model([o_inp], [classifier, bbox_regress, landmark_regress])
if os.path.exists(onet_weights_file):
    model.load_weights(onet_weights_file, by_name=True)
bbox_dense = model.get_layer('bbox1')
bbox_weight = bbox_dense.get_weights()
classifier_dense = model.get_layer('cls1')
cls_weight = classifier_dense.get_weights()
landmark_dense = model.get_layer('lmk1')
landmark_weight = landmark_dense.get_weights()

training_parts = 2
training_batches = 80

for i in range(training_parts):
   
    for i_train in range(training_batches):

        randx=random.choice([0,1,1,1,0,0,1])   # still need to run manually on each batch
        batch_size = 64
        print ('currently in training macro cycle: ',i_train)
    
        if 0 == randx:
            model = Model([o_inp], [classifier])
            model.get_layer('cls1').set_weights(cls_weight)
            model.compile(loss='binary_crossentropy', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_cls, one_hot_labels, batch_size=batch_size, epochs=1)
            classifier_dense = model.get_layer('cls1')
            cls_weight = classifier_dense.get_weights()
        if 1 == randx:
            model = Model([o_inp], [bbox_regress])
            model.get_layer('bbox1').set_weights(bbox_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_roi, roi_score, batch_size=batch_size, epochs=1)
            bbox_dense = model.get_layer('bbox1')
            bbox_weight = bbox_dense.get_weights()
        if 2 == randx:
            model = Model([o_inp], [landmark_regress])
            model.get_layer('lmk1').set_weights(landmark_weight)
            model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
            model.fit(ims_pts, pts_score, batch_size=batch_size, epochs=1)
            landmark_dense = model.get_layer('lmk1')
            landmark_weight = landmark_dense.get_weights()
    gc.collect()
    
    
    model = Model([o_inp], [classifier, bbox_regress, landmark_regress])
    model.get_layer('lmk1').set_weights(landmark_weight)
    model.get_layer('bbox1').set_weights(bbox_weight)
    model.get_layer('cls1').set_weights(cls_weight)
    model.save_weights(onet_weights_file)

