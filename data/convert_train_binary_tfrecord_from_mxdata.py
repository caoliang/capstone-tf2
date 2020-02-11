from absl import app, flags, logging
from absl.flags import FLAGS
import os
import io
import shutil
import imageio
import numpy as np
import cv2
import tqdm
import random
import tensorflow as tf
import mxnet as mx


flags.DEFINE_string('training_dataset_path', './data', 'training dataset path')
flags.DEFINE_string('dataset_path', 'faces_emore',
                    'path to dataset')
flags.DEFINE_string('output_path', 'ms1m_bin_in.tfrecord',
                    'path to ouput tfrecord')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(source_img, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/encoded': _bytes_feature(source_img),
               'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename) }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    idx_path = os.path.join(dataset_path, 'train.idx')
    bin_path = os.path.join(dataset_path, 'train.rec')
#    img_path = os.path.join(dataset_path, 'img')

#    if os.path.exists(img_path):
#        shutil.rmtree(img_path)

    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        mx_reader = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        first_rec = mx_reader.read_idx(0)
        first_header, _ = mx.recordio.unpack(first_rec)
        img_idx_list = list(range(1, int(first_header.label[0])))
        samples = []
        items_set = set()
        
        logging.info('Reading data list...')
        for i in tqdm.tqdm(img_idx_list):
            img_info = mx_reader.read_idx(i)
            header, _ = mx.recordio.unpack(img_info)
            img_label = int(header.label)
            
            #print(f'img_label: "{img_label}"')
            img_id = int(header.id)
            #img_folder =  os.path.join(img_path, f'{img_label}')
            #if not os.path.exists(img_folder):
            #    os.makedirs(img_folder)
            img_filename = f'{img_label}_{img_id}.jpg'.replace("\\", "/")
            #print(f'img_filename: "{img_filename}"')
            
            samples.append((img_id, img_label, img_filename))
            
            items_set.add(img_label)
            
#            if i % 50000 == 0:
#                print(f'Stop at total {i} pictures, {len(items_set)} classes')
#                break
        logging.info(f'Read total {i} pictures, {len(items_set)} classes')
        
#        random.shuffle(samples)
        
        logging.info('Writing tfrecord file...')
        img_num = 0
        for img_id, img_label, img_filename in tqdm.tqdm(samples):
            img_info = mx_reader.read_idx(img_id)
            header, img = mx.recordio.unpack(img_info)
            
            img = io.BytesIO(img)
            img = imageio.imread(img).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #img = cv2.resize(img, (112,112))
            img_str = cv2.imencode('.jpg', img)[1].tostring()
            
            #imageio.imwrite(img_filename, img)
            
            tf_example = make_example(source_img=img_str,
                                      source_id=img_label,
                                      filename=str.encode(img_filename))
            writer.write(tf_example.SerializeToString())
        
            img_num += 1
            
            if img_num % 10000 == 0:
                writer.flush()
#                print(f'Wrote total {img_num} pictures, {len(items_set)} classes')
#                break

        logging.info(f'Wrote total {img_num} pictures, {len(items_set)} classes')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
