"""The code to test training process for pnet"""
import sys
sys.path.append('..')

import tensorflow as tf
from mtcnn_tf2.mtcnn import train_net, PNet


def train_Pnet(training_data, base_lr, loss_weight,
               train_mode, num_epochs,
               load_model=False, load_filename=None,
               save_model=False, save_filename=None,
               num_iter_to_save=10000):

    train_net(Net=PNet,
              training_data=training_data,
              base_lr=base_lr,
              loss_weight=loss_weight,
              train_mode=train_mode,
              num_epochs=num_epochs,
              load_model=load_model,
              load_filename=load_filename,
              save_model=save_model,
              save_filename=save_filename,
              num_iter_to_save=num_iter_to_save)


if __name__ == '__main__':

    load_filename = '../data/mtcnn_training/pretrained/initial_weight_pnet.npy'
    save_filename = '../data/mtcnn_training/saved_model/pnet'
    training_data = ['../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_cls.tfrecords',
                     '../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_bbx.tfrecords']
    train_Pnet(training_data=training_data,
               base_lr=0.0001,
               loss_weight=[1.0, 0.5, 0.5],
               train_mode=2,
               num_epochs=[200, 200, 200],
               load_model=False,
               load_filename=load_filename,
               save_model=True,
               save_filename=save_filename,
               num_iter_to_save=100)
