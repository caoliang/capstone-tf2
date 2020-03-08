"""Generate tfrecords file for pnet, which has input size of 12*12*3."""

import sys
sys.path.append('../')

import random

import cv2
import tensorflow as tf
import numpy as np
import numpy.random as npr

from mtcnn_tf2.mtcnn_tools import view_bar, bytes_feature


def main():

    size = 12
    base_path = f'../data/mtcnn_training/native_{size}'

    with open(f'{base_path}/pos_{size}.txt', 'r') as f:
        pos = f.readlines()
    with open(f'{base_path}/neg_{size}.txt', 'r') as f:
        neg = f.readlines()
    with open(f'{base_path}/part_{size}.txt', 'r') as f:
        part = f.readlines()

    filename_cls = f'{base_path}/pnet_data_for_cls.tfrecords'
    examples = []
    writer = tf.io.TFRecordWriter(filename_cls)
    print(f'Writing {filename_cls}')
    
    print('\npos\nWriting')
    cls_total = 0
    cur_ = 0
    sum_ = len(pos)
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        #print(words)
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([0, 1], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
        
        if cur_ % 1000 == 0:
            random.shuffle(examples)
            for example in examples:
                writer.write(example.SerializeToString())
            writer.flush()
            examples.clear()
            
    cls_total += cur_

    print('\nneg\nWriting')
    cur_ = 0
    max_neg_size = 1000000
    neg_size = len(neg)
    neg_keep_size = min(max_neg_size, neg_size)
    neg_keep = npr.choice(neg_size, size=neg_keep_size, replace=False)
    sum_ = len(neg_keep)
    for i in neg_keep:
        line = neg[i]
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([1, 0], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
        
        if cur_ % 1000 == 0:
            random.shuffle(examples)
            for example in examples:
                writer.write(example.SerializeToString())
            writer.flush()
            examples.clear()
    
    cls_total += cur_
    cls_total += len(examples)
    
    print(f"Total cls records: {cls_total}")
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()
    examples.clear()

    filename_roi = f'{base_path}/pnet_data_for_bbx.tfrecords'
    writer = tf.io.TFRecordWriter(filename_roi)
    print(f'Writing {filename_roi}')
    
    roi_total = 0
    print('\npos\nWriting')
    cur_ = 0
    sum_ = len(pos)
    examples = []
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
        
        if cur_ % 1000 == 0:
            random.shuffle(examples)
            for example in examples:
                writer.write(example.SerializeToString())
            writer.flush()
            examples.clear()

    roi_total += cur_

    print('\npart\nWriting')
    cur_ = 0
    max_part_size = 300000
    part_size = len(part)
    part_keep_size = min(max_part_size, part_size)
    part_keep = npr.choice(part_size, size=part_keep_size, replace=False)
    sum_ = len(part_keep)
    for i in part_keep:
        view_bar(cur_, sum_)
        line = part[i]
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
        
        if cur_ % 1000 == 0:
            random.shuffle(examples)
            for example in examples:
                writer.write(example.SerializeToString())
            writer.flush()
            examples.clear()

    roi_total += cur_
    roi_total += len(examples)    
        
    print(f'Total roi records: {roi_total}')
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()