import cv2
import sys
import argparse
import numpy as np
from mtcnn_tf2 import face_mtcnn, detect_face_lib

def main(args):
    try:
#        onet = face_mtcnn.create_onet()
#        print(onet)
        
        mtcnn_nets = detect_face_lib.create_mtcnn_nets()
        print(mtcnn_nets)
    
        image = cv2.cvtColor(cv2.imread("cl.jpg"), cv2.COLOR_BGR2RGB)
        image_features = detect_face_lib.detect_face(image, mtcnn_nets)
        print(image_features)
#        result = detector.detect_faces(image)
        
    #    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    #    bounding_box = result[0]['box']
    #    keypoints = result[0]['keypoints']
    #    
    #    cv2.rectangle(image,
    #                  (bounding_box[0], bounding_box[1]),
    #                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
    #                  (0,155,255),
    #                  2)
    #    
    #    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    #    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    #    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    #    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    #    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #    
    #    cv2.imwrite("cl_detected.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
#        print(result)
    except Exception as ex:
        print(ex)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='./data/faces_source', help='Directory with unaligned images.')
    parser.add_argument('--output_dir', type=str, default='./data/faces_store', help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))