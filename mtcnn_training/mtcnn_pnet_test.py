"""The code to test training process for pnet"""
import sys
sys.path.append('..')

from mtcnn_tf2.mtcnn import train_net, PNet

def train_Pnet(training_data, base_lr,
               num_epochs, save_filename=None):

    train_net(Net=PNet,
              training_data=training_data,
              base_lr=base_lr,
              num_epochs=num_epochs,
              save_filename=save_filename)


if __name__ == '__main__':

    model_filename = '../data/mtcnn_training/saved_model/pnet.hdf5'
    training_data = ['../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_cls.tfrecords',
                     '../data/mtcnn_training/native_12/' + 
                     'pnet_data_for_bbx.tfrecords']
    train_Pnet(training_data=training_data,
               base_lr=0.0001,
               num_epochs=1,
               save_filename=model_filename)
