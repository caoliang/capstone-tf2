# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:38:12 2020

@author: cl
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Tfrecords test codes

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tf_file(tf_file):
    line = '../data/mtcnn_training/native_12/positive/0 1 0.03 0.02 -0.15 0.01'
    
    words = line.split()
    
    image_file_name = words[0]+'.jpg'
    print(image_file_name)
    
    im = cv2.imread(image_file_name)
    
    h, w, ch = im.shape
    print(f"h={h}, w={w}, ch={ch}")
    if h != 12 or w != 12:
        print('resize')
        im = cv2.resize(im, (12, 12))
        
    im = im.astype('uint8')
    label = np.array([0, 1], dtype='float32')
    label_raw = label.tostring()
    image_raw = im.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': bytes_feature(label_raw),
        'image_raw': bytes_feature(image_raw)}))
    
    writer = tf.io.TFRecordWriter(tf_path)
    writer.write(example.SerializeToString())
    writer.close()


def parse_dataset_callback(serialized_data):
    raw_desc = {
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),}
    
    return tf.io.parse_single_example(serialized_data, raw_desc)

def read_tf_file(tf_file):
    raw_dataset = tf.data.TFRecordDataset(tf_file)
    parsed_dataset = raw_dataset.map(parse_dataset_callback)
    print(f"parsed_dataset: {parsed_dataset}")
    
    for data_line in parsed_dataset:
        label_raw = tf.io.decode_raw(data_line['label_raw'], tf.float32)
        print(f"label_raw: {label_raw}")
        image_raw = tf.io.decode_raw(data_line['image_raw'], tf.uint8)
        print(f"image_raw: {image_raw}, shape: {image_raw.shape}")
        image_raw_reshape = np.reshape(image_raw, (12, 12, 3))
        print(f"image_raw_reshape: {image_raw_reshape}, " + 
              f"shape: {image_raw_reshape.shape}")
        image = tf.cast(image_raw, tf.float32)
        print(f"image: {image}")
        
        plt.imshow(image_raw_reshape)
    return parsed_dataset

tf_dir = '../data/mtcnn_training/native_12'
tf_name = "test.tfrecords"
tf_path = os.path.join(tf_dir, tf_name)

#write_tf_file(tf_path)
#print(f"Completed to write tfrecords test file '{tf_path}'")

dataset = read_tf_file(tf_path)
print(f"Completed to read tfrecords test file '{tf_path}'")
