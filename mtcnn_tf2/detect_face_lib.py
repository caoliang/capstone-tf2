import cv2
import numpy as np
import time
import logging
from mtcnn_tf2.face_mtcnn import build_mtcnn_nets
import mtcnn_tf2.tools_matrix as tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """

    def __init__(self, pad_result: tuple = None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = \
            self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = \
            pad_result


def create_mtcnn_nets():
    pnet, rnet, onet = build_mtcnn_nets()
    return [pnet, rnet, onet]


def compute_scale_pyramid(m, min_layer, scale_factor):
    scales = []
    factor_count = 0

    while min_layer >= 12:
        scales += [m * np.power(scale_factor, factor_count)]
        min_layer = min_layer * scale_factor
        factor_count += 1

    return scales

def detect_faces(image = None, mtcnn_nets = None, min_face_size = 20, 
                 scale_factor = 0.709, steps_threshold = [0.6, 0.7, 0.7]):
    """
    Detects bounding boxes from the specified image.
    :param img: image to process
    :return: list containing all the bounding boxes detected with their keypoints.
    """    
    total_boxes, points = detect_faces_features(image, mtcnn_nets, 
                                                min_face_size, scale_factor,
                                                steps_threshold)
    binding_boxes = extract_binding_box(total_boxes, points)
    
    return binding_boxes
    

def detect_faces_features(image = None, mtcnn_nets = None, 
                          min_face_size = 20, scale_factor = 0.709,
                          steps_threshold = [0.6, 0.7, 0.7]):
    if image is None or not hasattr(image, "shape"):
        raise Exception("Face image is invalid")
    
    if mtcnn_nets is None:
        raise Exception("Cannot get mtcnn nets")
        
    height, width, _ = image.shape
    logger.info("image.shape {0}".format(image.shape))
    stage_status = StageStatus(width=width, height=height)
    
    m = 12 / min_face_size
    min_layer = np.amin([height, width]) * m
    
    scales = compute_scale_pyramid(m, min_layer, scale_factor)
    
    stages = [__stage1, __stage2, __stage3]
    
    result = [scales, stage_status]
    
    # We pipe here each of the stages
    for index in range(len(stages)):
        result = stages[index](image, steps_threshold[index], 
                       mtcnn_nets[index], result[0], result[1])
    
    [total_boxes, points] = result
    
    return total_boxes, points

def extract_binding_box(all_boxes, all_points):
    bounding_boxes = []
    
    for bounding_box, keypoints in zip(all_boxes, all_points.T):
            bounding_boxes.append({
                'box': [max(0, int(bounding_box[0])), max(0, int(bounding_box[1])),
                        int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])],
                'confidence': bounding_box[-1],
                'keypoints': {
                    'left_eye': (int(keypoints[0]), int(keypoints[5])),
                    'right_eye': (int(keypoints[1]), int(keypoints[6])),
                    'nose': (int(keypoints[2]), int(keypoints[7])),
                    'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                    'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                }
            })
    
    return bounding_boxes

def __scale_image(image, scale: float):
    """
    Scales the image to a given scale.
    :param image:
    :param scale:
    :return:
    """
    height, width, _ = image.shape

    width_scaled = int(np.ceil(width * scale))
    height_scaled = int(np.ceil(height * scale))

    im_data = cv2.resize(image, (width_scaled, height_scaled), 
                         interpolation=cv2.INTER_AREA)

    # Normalize the image's pixels
    im_data_normalized = (im_data - 127.5) * 0.0078125

    return im_data_normalized


def __generate_bounding_box(imap, reg, scale, box_threshold):

    # use heatmap to generate bounding boxes
    stride = 2
    cellsize = 12
    
    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    
    y, x = np.where(imap >= box_threshold)
    
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], 
                                  dx2[(y, x)], dy2[(y, x)]]))
    
    if reg.size == 0:
        reg = np.empty(shape=(0, 3))
    
    bb = np.transpose(np.vstack([y, x]))
    
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    
    return boundingbox, reg


def __nms(boxes, threshold, method):
    """
    Non Maximum Suppression.

    :param boxes: np array with bounding boxes.
    :param threshold:
    :param method: NMS method to apply. Available values ('Min', 'Union')
    :return:
    """
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_s = np.argsort(s)

    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while sorted_s.size > 0:
        i = sorted_s[-1]
        pick[counter] = i
        counter += 1
        idx = sorted_s[0:-1]

        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        sorted_s = sorted_s[np.where(o <= threshold)]

    pick = pick[0:counter]

    return pick


def __pad(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]
    
    dx = np.ones(numbox, dtype=np.int32)
    dy = np.ones(numbox, dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)
    
    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)
    
    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w
    
    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h
    
    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1
    
    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def __rerec(bbox):
    # convert bbox to square
    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    max_side_length = np.maximum(width, height)
    bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
    bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
    bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
    return bbox

def __bbreg(boundingbox, reg):
    # calibrate bounding boxes
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
    
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def __stage1(img, step_threshold, mtcnn_net, scales, stage_status):
    """
    First stage of the MTCNN.
    :param image:
    :param mtcnn_net:
    :param scales:
    :param stage_status:
    :return:
    """
    total_boxes = np.empty((0, 9))
    status = stage_status
    
    for scale in scales:
        scaled_image = __scale_image(img, scale)
    
        img_x = np.expand_dims(scaled_image, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
    
        p_classifier, p_bbox_regress = mtcnn_net.predict(img_y)
    
        p_bbox_regress = np.transpose(p_bbox_regress, (0, 2, 1, 3))
        p_classifier = np.transpose(p_classifier, (0, 2, 1, 3))
    
        boxes, _ = __generate_bounding_box(p_classifier[0, :, :, 1].copy(),
                                           p_bbox_regress[0, :, :, :].copy(), 
                                           scale, step_threshold)
    
        # inter-scale nms
        pick = __nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)
    
    numboxes = total_boxes.shape[0]
    
    if numboxes > 0:
        pick = __nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
    
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
    
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
    
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, 
                                              total_boxes[:, 4]]))
        total_boxes = __rerec(total_boxes.copy())
    
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        status = StageStatus(__pad(total_boxes.copy(), stage_status.width, 
                                   stage_status.height),
                             width=stage_status.width, 
                             height=stage_status.height)
    
    return total_boxes, status


def __stage2(img, step_threshold, mtcnn_net, total_boxes, stage_status):
    """
    Second stage of the MTCNN.
    :param img:
    :param step_threshold:
    :param total_boxes:
    :param stage_status:
    :return:
    """

    num_boxes = total_boxes.shape[0]
    if num_boxes == 0:
        return total_boxes, stage_status

    # second stage
    tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

    for k in range(0, num_boxes):
        tmp = np.zeros((int(stage_status.tmph[k]), 
                        int(stage_status.tmpw[k]), 3))

        tmp[stage_status.dy[k] - 1:stage_status.edy[k], 
            stage_status.dx[k] - 1:stage_status.edx[k], :] = \
            img[stage_status.y[k] - 1:stage_status.ey[k],
                stage_status.x[k] - 1:stage_status.ex[k], :]

        if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 
            and tmp.shape[1] == 0):
            
            tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), 
                                  interpolation=cv2.INTER_AREA)

        else:
            return np.empty(shape=(0,)), stage_status

    tempimg = (tempimg - 127.5) * 0.0078125
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

    r_classifier, r_bbox_regress = mtcnn_net.predict(tempimg1)

    r_bbox_regress = np.transpose(r_bbox_regress)
    r_classifier = np.transpose(r_classifier)

    score = r_classifier[1, :]

    ipass = np.where(score > step_threshold)

    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), 
                             np.expand_dims(score[ipass].copy(), 1)])

    mv = r_bbox_regress[:, ipass[0]]

    if total_boxes.shape[0] > 0:
        pick = __nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        total_boxes = __bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
        total_boxes = __rerec(total_boxes.copy())

    return total_boxes, stage_status


def __stage3(img, step_threshold, mtcnn_net, total_boxes, stage_status):
    """
    Third stage of the MTCNN.

    :param img:
    :param total_boxes:
    :param stage_status:
    :return:
    """
    num_boxes = total_boxes.shape[0]
    if num_boxes == 0:
        return total_boxes, np.empty(shape=(0,))

    total_boxes = np.fix(total_boxes).astype(np.int32)

    status = StageStatus(__pad(total_boxes.copy(), stage_status.width, 
                                    stage_status.height),
                         width=stage_status.width, height=stage_status.height)

    tempimg = np.zeros((48, 48, 3, num_boxes))

    for k in range(0, num_boxes):

        tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

        tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
            img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

        if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
            tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
        else:
            return np.empty(shape=(0,)), np.empty(shape=(0,))

    tempimg = (tempimg - 127.5) * 0.0078125
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

    o_classifier, o_bbox_regress, o_landmark_regress = mtcnn_net.predict(tempimg1)
    o_bbox_regress = np.transpose(o_bbox_regress)
    o_landmark_regress = np.transpose(o_landmark_regress)
    o_classifier = np.transpose(o_classifier)

    score = o_classifier[1, :]

    points = o_landmark_regress

    ipass = np.where(score > step_threshold)

    points = points[:, ipass[0]]

    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), 
                             np.expand_dims(score[ipass].copy(), 1)])

    mv = o_bbox_regress[:, ipass[0]]

    w = total_boxes[:, 2] - total_boxes[:, 0] + 1
    h = total_boxes[:, 3] - total_boxes[:, 1] + 1

    points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + \
                     np.tile(total_boxes[:, 0], (5, 1)) - 1
    points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + \
                      np.tile(total_boxes[:, 1], (5, 1)) - 1

    if total_boxes.shape[0] > 0:
        total_boxes = __bbreg(total_boxes.copy(), np.transpose(mv))
        pick = __nms(total_boxes.copy(), 0.7, 'Min')
        total_boxes = total_boxes[pick, :]
        points = points[:, pick]

    return total_boxes, points