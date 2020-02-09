import logging
from multiprocessing.pool import Pool

import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Utils import Data

log = logging.getLogger('DataSources')


dlib_detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor('./detect_face/dlib_shape_predictor_68_face_landmarks.dat')

except:
    log.warning('Failed to load dlib face landmarks ds - landmark detections will not work!')

DNN = "TF"
if DNN == "CAFFE":
    modelFile = "detect_face/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "detect_face/deploy.prototxt"
    cv_dnn_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "detect_face/opencv_face_detector_uint8.pb"
    configFile = "detect_face/opencv_face_detector.pbtxt"
    cv_dnn_net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

DEBUG = False


def detect_face_cv_dnn(image):

    image_mat = dlib.load_rgb_image(image)
    frameHeight, frameWidth = image_mat.shape[0], image_mat.shape[1]

    blob = cv2.dnn.blobFromImage(image_mat)

    cv_dnn_net.setInput(blob)
    detections = cv_dnn_net.forward()  # returns [1, 1, 200, 7] -> 200 detections, 3-6 -> bbox, 2 -> confidence

    if len(detections[0][0]) == 0:
        log.info('detect_face_cv_dnn:: ERROR - failed to find face bbox for %s' % image)
        return None

    confidence_scores = np.array(detections[0][0]).transpose()[2]
    max_confidence_i = np.argmax(confidence_scores)

    x, y = int(detections[0, 0, max_confidence_i, 3] * frameWidth * 0.95), int(detections[0, 0, max_confidence_i, 4] * frameHeight)
    w, h = int(detections[0, 0, max_confidence_i, 5] * frameWidth * 0.95) - x, int(detections[0, 0, max_confidence_i, 6] * frameHeight * 1.02) - y

    if DEBUG:
        cv2.rectangle(image_mat, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.figure()
        plt.imshow(image_mat, cmap='gray', interpolation='nearest')
        plt.show()

    if confidence_scores[max_confidence_i] < 0.51:
        log.info('detect_face_cv_dnn:: ERROR - found face bbox for %s with very low confidence %s' % (image, confidence_scores[max_confidence_i]))
        return None

    return np.array([x, y, w, h])


def detect_face_dlib(data: Data) -> Data:
    image_mat = dlib.load_rgb_image(data.image)
    faces, scores, idx = dlib_detector.run(image_mat, 1, -1)
    # log.info(faces)

    if len(faces) > 1:
        # log.info('*************************** TOO MANY FACES - get the one with max score !!!!')
        index = 0
        max_score = scores[0]
        for i, score in enumerate(scores):
            if score > max_score:
                max_score = score
                index = i
        # shape = predictor(image_gray, faces[0])
        x = faces[index].left()
        y = faces[index].top()
        w = faces[index].right() - x
        h = faces[index].bottom() - y
    elif len(faces) > 0:
        x = faces[0].left()
        y = faces[0].top()
        w = faces[0].right() - x
        h = faces[0].bottom() - y
    else:
        log.error('detect_face_dlib:: ERROR - failed to find face bbox for %s' % data.image)
        return data

    data.bbox = np.array([x, y, w, h])
    return data


def detect_face_landmarks_dlib(data: Data) -> Data:
    try:
        image = cv2.imread(data.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = data.bbox
        dlib_rect = dlib.rectangle(x, y, x+ w, y + h)
        landmarks = []
        predicted_landmarks = predictor(image, dlib_rect)
        for i in range(68):
            landmarks.append(np.array([predicted_landmarks.part(i).x, predicted_landmarks.part(i).y], dtype=np.float32))
        data.landmarks_2d = np.array(landmarks, dtype=np.float32)
    except:
        log.warning('did not find landmarks to image %s ' % data.image)
    return data


def detect_face_multithread(data: [Data], fn=detect_face_dlib, threads=5) -> [Data]:
    log.info('detect_face_multithread::started::%s::%s images' % (fn.__name__, len(data)))
    batch_size = 50
    p = Pool(processes=threads)
    data_: [Data] = []
    for i in range(int(np.floor(len(data) / batch_size))):
        log.info('detect_face_multithread:: %s/%s' % (i*batch_size, len(data)))
        data_ += p.map(fn, data[i * batch_size: (i + 1) * batch_size])

    if len(data) % batch_size > 0:
        data_ += p.map(fn, data[int(np.floor(len(data) / batch_size) * batch_size):])

    p.close()
    p.join()

    return data_


def get_face_bb_multithread(data: [Data], threads=5) -> [Data]:
    log.info('get_face_bb_multithread::started:: %s images' % (len(data)))
    batch_size = 50
    p = Pool(processes=threads)
    data_with_bbox: [Data] = []
    for i in range(int(np.floor(len(data) / batch_size))):
        log.info('get_face_bb_multithread:: %s/%s' % (i*batch_size, len(data)))
        data_with_bbox += p.map(get_face_bb, data[i * batch_size: (i+1) * batch_size])

    if len(data) % batch_size > 0:
        data_with_bbox += p.map(get_face_bb, data[int(np.floor(len(data) / batch_size) * batch_size):])

    p.close()
    p.join()

    return data_with_bbox


def get_face_bboxes(data: [Data]) -> [Data]:
    data = [get_face_bb(data_elm) for data_elm in data]
    return data


def get_face_bb(data: Data) -> Data:
    data.bbox = get_face_bb2(data.landmarks_2d)
    return data


def get_face_bb2(landmarks_2d) -> np.array:
    bounding_rect = cv2.boundingRect(landmarks_2d.astype(np.int))
    x, y, w, h = bounding_rect
    return np.array([x, y, w, h])


if __name__ == '__main__':
    image = '../datasets/300W_LP/HELEN/HELEN_100032540_1_11.jpg'
    # image = '../datasets/300W_LP/HELEN/HELEN_100032540_1_17.jpg'
    image = '../augmented/valid_set_naive/image_03599_2_aug.png'
    image = '../augmented/valid_set_naive/image_00232_13_aug.png'
    image = '../augmented/valid_set_naive/image_00121_41_aug.png'
    # log.info(detect_face_cv_dnn(image))
    log.info(detect_face_multithread([image], fn=detect_face_cv_dnn))
