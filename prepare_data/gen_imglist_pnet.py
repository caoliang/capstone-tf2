import sys
sys.path.append('..')

import logging as pylog
from absl import app, flags
from absl import logging as alog
from absl.flags import FLAGS

import numpy as np
import numpy.random as npr
import os

from modules.utils import load_yaml

flags.DEFINE_string('train_cfg_path', '../configs/mtcnn_pnet.yaml', 
                    'MTCNN pnet configuration file path')

flags.DEFINE_bool('log_to_file', True, 'Whether log to file')
flags.DEFINE_string('log_file', 'gen_imglist_pnet', 'log filename')
flags.DEFINE_string('log_dir_path', '../logs/prepare_data', 
                    'log file directory path')

def main(_):
    if FLAGS.log_to_file: 
        alog.get_absl_handler().use_absl_log_file(FLAGS.log_file, 
                         FLAGS.log_dir_path)
    alog.set_verbosity(alog.DEBUG)
    
    cfg = load_yaml(FLAGS.train_cfg_path)

    training_data_dir = cfg['training_data_dir']
    #anno_file = os.path.join(training_data_dir, "anno.txt")
    
    size = 12
    
    if size == 12:
        net = "PNet"
    elif size == 24:
        net = "RNet"
    elif size == 48:
        net = "ONet"
    
    pos_file_path = os.path.join(training_data_dir, 
                                 'pnet_%s/pos_%s.txt' % (size, size)) 
    alog.info("Read pos_file_path '%s'", pos_file_path)
    with open(pos_file_path, 'r') as f:
        pos = f.readlines()
    
    neg_file_path = os.path.join(training_data_dir, 
                                 'pnet_%s/neg_%s.txt' % (size, size))
    alog.info("Read neg_file_path '%s'", neg_file_path)
    with open(neg_file_path, 'r') as f:
        neg = f.readlines()
    
    part_file_path = os.path.join(training_data_dir, 
                                  'pnet_%s/part_%s.txt' % (size, size))
    alog.info("Read part_file_path '%s'", part_file_path)
    with open(part_file_path, 'r') as f:
        part = f.readlines()
    
    landmark_file_path = os.path.join(training_data_dir, 
                                'pnet_%s/landmark_%s_aug.txt' % (size,size))
    alog.info("Read landmark_file_path '%s'", landmark_file_path)
    with open(landmark_file_path, 'r') as f:
        landmark = f.readlines()
        
    dir_path = os.path.join(training_data_dir, 'imglists')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
        os.makedirs(os.path.join(dir_path, "%s" %(net)))
    
    pnet_imglist_file_path = os.path.join(dir_path, "%s" %(net), 
                                          "train_%s_landmark.txt" % (net))

    alog.info("Read pnet_imglist_file_path '%s'", pnet_imglist_file_path)
    
    count_img = 0
    
    with open(pnet_imglist_file_path, "w") as f:
        count_img += 1
        
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        #base_num = min(nums)
        base_num = 250000
        
        alog.info(f"neg: {len(neg)}, pos: {len(pos)}, "
                  + f"part: {len(part)}, landmark: {len(landmark)}, "
                  + f"base_num: {base_num}")
    
        #shuffle the order of the initial data
        #if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        landmark_keep = npr.choice(len(landmark), size=base_num, replace=True)
        
        alog.info(f"neg_keep: {len(neg_keep)}, pos_keep: {len(pos_keep)}, "
                  + f"part_keep: {len(part_keep)}, "
                  + f"landmark_keep: {len(landmark_keep)}")
    
        # write the data according to the shuffled order
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for i in landmark_keep:
            f.write(landmark[i])

    alog.info('Completed')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass