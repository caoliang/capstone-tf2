import sys
# For importing mtcnn_tf2 folder
sys.path.append('..')

import numpy as np
import cv2
import os
import numpy.random as npr
from mtcnn_tf2.face_mtcnn import build_pnet
from mtcnn_tf2.mtcnn_tools import IoU, detect_face_12net, view_bar
import _pickle as pickle

image_size = 24
net_size = 12

anno_file = r"..\data\WIDER_train\wider_face_train.txt"
im_dir = r"..\data\WIDER_train\images"
pos_save_dir = r"..\data\mtcnn_training\{0:d}net\positive_hard".format(net_size)
part_save_dir = r"..\data\mtcnn_training\{0:d}net\part_hard".format(net_size)
neg_save_dir = r'..\data\mtcnn_training\{0:d}net\negative_hard'.format(net_size)
save_dir = r'..\data\mtcnn_training\{0:d}net'.format(net_size)

net_weights_file = r'..\data\mtcnn_training\model{0:d}.h5'.format(net_size)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#if not os.path.exists(pos_save_dir):
#    os.mkdir(pos_save_dir)
#if not os.path.exists(part_save_dir):
#    os.mkdir(part_save_dir)
#if not os.path.exists(neg_save_dir):
#    os.mkdir(neg_save_dir)

# Load pnet
pnet_model = build_pnet(net_weights_file)
    
f1 = open(os.path.join(save_dir, 'hard_pos_{0:d}.txt'.format(image_size)), 'w')
f2 = open(os.path.join(save_dir, 'hard_neg_{0:d}.txt'.format(image_size)), 'w')
f3 = open(os.path.join(save_dir, 'hard_part_{0:d}.txt'.format(image_size)), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("{:d} pics in total".format(num))

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
cls_list = []
roi_list = []

pos_cls_list = []
pos_roi_list = []
neg_cls_list = []
part_roi_list = []

minsize = 20
factor = 0.709
threshold = 0.6

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]))
    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    rectangles = detect_face_12net(img, minsize, pnet_model,
                                   threshold, factor)
    
    print ("rectangles: " + str(rectangles))

    idx += 1
    if idx % 1000 == 0:
        print (idx, "images done")

    #height, width, channel = img.shape

    for box in rectangles:
        lis = box.astype(np.int32)
        mask = lis < 0
        lis[mask] = 0
        x_left, y_top, x_right, y_bottom, _ = lis
        crop_w = x_right - x_left + 1
        crop_h = y_bottom - y_top + 1
        # ignore box that is too small or beyond image border
        if crop_w < image_size or crop_h < image_size:
            continue
    
        Iou = IoU(box, gts)
        cropped_im = img[y_top: y_bottom+1, x_left: x_right+1]
        resized_im = cv2.resize(cropped_im,
                                (image_size, image_size),
                                interpolation=cv2.INTER_LINEAR)
    
        # save negative images and write label
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
            f2.write(r"xx\negative\%s"%n_idx + ' 0\n')
            
            im = resized_im
            h, w, ch = resized_im.shape
            if h != image_size or w != image_size:
                im = cv2.resize(im, (image_size, image_size))
            im = np.swapaxes(im, 0, 2)
            im = (im - 127.5) / 127.5
            label = 0
            roi = [-1, -1, -1, -1]
            pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            neg_cls_list.append([im, label, roi])
            
            n_idx += 1
        else:
            # find gt_box with the highest iou
            idx = np.argmax(Iou)
            assigned_gt = gts[idx]
            x1, y1, x2, y2 = assigned_gt
    
            # compute bbox reg label
            offset_x1 = (x1 - x_left) / float(crop_w)
            offset_y1 = (y1 - y_top) / float(crop_h)
            offset_x2 = (x2 - x_right) / float(crop_w)
            offset_y2 = (y2 - y_bottom) / float(crop_h)
    
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                f1.write(r"xx\positive\%s"%p_idx + 
                         ' 1 %.2f %.2f %.2f %.2f\n'%
                         (offset_x1, offset_y1, offset_x2, offset_y2))

                im = resized_im
                h, w, ch = resized_im.shape
                if h != image_size or w != image_size:
                    im = cv2.resize(im, (image_size, image_size))
                im = np.swapaxes(im, 0, 2)
                im = (im - 127.5) / 127.5
                label = 1
                roi = [-1, -1, -1, -1]
                pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                pos_cls_list.append([im, label, roi])

                roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                pos_roi_list.append([im, label, roi])
    
                p_idx += 1
            elif np.max(Iou) >= 0.4:
                save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                f3.write(r"xx\part\%s"%d_idx + 
                         ' -1 %.2f %.2f %.2f %.2f\n'%
                         (offset_x1, offset_y1, offset_x2, offset_y2))
                
                im = resized_im
                h, w, ch = resized_im.shape
                if h != image_size or w != image_size:
                    im = cv2.resize(im, (image_size, image_size))
                im = np.swapaxes(im, 0, 2)
                # im -= 127
                im = (im - 127.5) / 127.5 #  it is wrong in original code
                label = -1
                roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                part_roi_list.append([im, label, roi])
                
                d_idx += 1

    box_idx += 1
    if (idx % 500 == 0 or idx == num):
        print ("{:d} images done, positive: {:d} part: {:d} negative: {:d}".format(idx, p_idx, d_idx, n_idx))



f1.close()
f2.close()
f3.close()

# limit the amount of negative sample to keep
neg_keep = npr.choice(len(neg_cls_list), size=p_idx*3, replace=False)  
# limit the amount of part sample to keep
part_keep = npr.choice(len(part_roi_list), size=p_idx * 1, replace=False)

for i in neg_keep:
    cls_list.append(neg_cls_list[i])

cls_list.extend(pos_cls_list)

for i in part_keep:
    roi_list.append(part_roi_list[i])
roi_list.extend(pos_roi_list)


cls_list_file = r"..\data\mtcnn_training\{0:d}net\hard_cls.imdb".format(net_size)

fid = open(cls_list_file,'wb')
pickle.dump(cls_list, fid)
fid.close()

cls_list = []


roi_list_file = r"..\data\mtcnn_training\{0:d}net\hard_roi.imdb".format(net_size)

fid = open(roi_list_file, 'wb')
pickle.dump(roi_list, fid)
fid.close()

roi_list = []


