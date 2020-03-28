# -*- coding: utf-8 -*-
"""
Generate train_imglist_align.txt to combine list_bbox_celeba.txt
and list_landmarks_align_celeba.txt content in the format:
image_file bbox landmark
bbox: left, right, top, bottom
landmark: (x1, y1), (x2, y2), ...

Created on Mon Mar 23 22:12:51 2020

@author: cl
"""

import sys
sys.path.append('..')

from absl import app, flags
from absl import logging as alog
from absl.flags import FLAGS

flags.DEFINE_bool('log_to_file', True, 'Whether log to file')
flags.DEFINE_string('log_file', 'gen_train_celeba_imglist', 'log filename')
flags.DEFINE_string('log_dir_path', '../logs/prepare_data', 
                    'log file directory path')

def main(_):
    
    if FLAGS.log_to_file: 
        alog.get_absl_handler().use_absl_log_file(FLAGS.log_file, 
                         FLAGS.log_dir_path)
    alog.set_verbosity(alog.DEBUG)
    alog.info('Started')

    list_bbox_file = "../data/celebA/Anno/list_bbox_celeba.txt"
    list_landmark_file = "../data/celebA/Anno/list_landmarks_align_celeba.txt"
    train_imglist_file = "../data/celebA/Anno/train_imglist_align.txt"
    
    with open(list_bbox_file, 'r') as infile:
        list_bbox = infile.readlines()
    with open(list_landmark_file, 'r') as infile:
        list_landmark = infile.readlines()
    
    # Content starts from 3rd line, so skip 2 lines    
    num_list_bbox = len(list_bbox) - 2
    num_list_landmark = len(list_landmark) - 2
    alog.info(f"{num_list_bbox} in bbox and {num_list_landmark} in landmark")

    if num_list_bbox != num_list_landmark:
        alog.error("File error! bbox lines not same as landmark")
        return

    with open(train_imglist_file, 'w') as outfile:
        for num_line in range(2, num_list_bbox):
            bbox_pos_list = list(filter(None, 
                                    list_bbox[num_line].strip().split(' ')))
            landmark_pos_list = list(filter(None, 
                                list_landmark[num_line].strip().split(' ')))
            
            if bbox_pos_list[0] != landmark_pos_list[0]:
                alog.error("Image path Error! " + 
                           "'{bbox_pos_list[0]}' != '{landmark_pos_list[0]}'")
                break
            
            # image_id x_1 y_1 width height
            img_path = bbox_pos_list[0]
            # Converto x1, x2, y1, y2 (left, right, top, bottom)
            if num_line <= 2:
                alog.debug(f"bbox_pos_list[1:]: {bbox_pos_list[1:]}")
                alog.debug(f"landmark_pos_list[1:]: {landmark_pos_list[1:]}")
                
            bbox_pos_int = list(map(int, bbox_pos_list[1:]))
            bbox_pos = [bbox_pos_int[0], 
                        bbox_pos_int[0] + bbox_pos_int[2],
                        bbox_pos_int[1], 
                        bbox_pos_int[1] + bbox_pos_int[3]]
            bbox_str = " ".join(list(map(str, bbox_pos)))
            landmark_str = " ".join(landmark_pos_list[1:])
            
            imglist_line = f"{img_path} {bbox_str} {landmark_str}\n"
            outfile.write(imglist_line)
            
            if num_line % 100 == 0:
                alog.debug(f"Line: '{imglist_line}'")
                alog.info(f"Completed {num_line} lines")
            
            
    alog.info('Completed')       

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass