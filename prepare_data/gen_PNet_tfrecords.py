#coding:utf-8
import sys
sys.path.append('..')

import logging as pylog
from absl import app, flags
from absl import logging as alog
from absl.flags import FLAGS

import os
from os.path import join, exists
import random
import sys
import time

import tensorflow as tf
from tensorflow.compat.v1.python_io import TFRecordWriter
from prepare_data.tfrecord_utils import _process_image_withoutcoder
from prepare_data.tfrecord_utils import _convert_to_example_simple

from modules.utils import load_yaml

flags.DEFINE_string('train_cfg_path', '../configs/mtcnn_pnet.yaml', 
                    'MTCNN pnet configuration file path')

flags.DEFINE_bool('log_to_file', True, 'Whether log to file')
flags.DEFINE_string('log_file', 'gen_PNet_tfrecords', 'log filename')
flags.DEFINE_string('log_dir_path', '../logs/prepare_data', 
                    'log file directory path')


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      filename: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    #print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def main(_):
    if FLAGS.log_to_file: 
        alog.get_absl_handler().use_absl_log_file(FLAGS.log_file, 
                         FLAGS.log_dir_path)
    alog.set_verbosity(alog.DEBUG)
    
    cfg = load_yaml(FLAGS.train_cfg_path)
    
    training_data_dir = cfg['training_data_dir']
    alog.info(f"training_data_dir: {training_data_dir}")
    net = 'PNet'
    pnet_out_dir = join(training_data_dir, 'imglists', net)
    dataset_in_dir = pnet_out_dir
    #output_directory = '../../DATA/imglists/PNet'
    alog.info('pnet_out_dir: %s', pnet_out_dir)
    run(dataset_in_dir, net, pnet_out_dir, shuffling=True)


def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = join(output_dir, "train_%s_landmark.tfrecord" % net)
    if shuffling:
        tf_filename = join(output_dir, 
                                   "train_%s_landmark_shuffle.tfrecord" % net)
        
    if exists(tf_filename):
        print('Dataset files already exist. Delete them.')
        os.remove(tf_filename)
        
    # GET Dataset, and shuffling.
    data_filename = join(dataset_dir, 'train_%s_landmark.txt' % net)
    dataset = get_dataset(data_filename, net=net)
    # filenames = dataset['filename']
    if shuffling:
        #random.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    alog.info('started to write tfrecord')
    with TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
            if (i+1) % 100 == 0:
                alog.debug('>> %d/%d images has been converted' % (i+1, len(dataset)))
                tfrecord_writer.flush()
            
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    alog.info('Finished converting the PNet MTCNN dataset!')


def get_dataset(data_filename, net='PNet'):
    imagelist = open(data_filename, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        #print(data_example['filename'])
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
            
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
   try:
       app.run(main)
   except SystemExit:
       pass
