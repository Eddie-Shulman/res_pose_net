import csv
import logging

import DataSources
import Utils
from Utils import Data
from detect_face import DetectFace
import Model
import numpy as np
import cv2

from Settings import *

log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)
# log.addHandler(ch)


def validate_predictions(predicted_poses, data: [Data], is_6pos=False):

    theta_error = 0.
    all_deg_error = 0
    total_samples_compared = 0

    for i, predicted_pose in enumerate(predicted_poses):
        if is_6pos:
            rx_p, ry_p, rz_p, tx_p, ty_p, tz_p = predicted_pose
        else:
            rx_p, ry_p, rz_p = predicted_pose

        if data[i].pose is not None:
            rx_v, ry_v, rz_v, tx_l, ty_l, tz_l = data[i].pose
        else:
            log.warning('image %s doesnt have a pose! skipping...')
            continue

        euc_distance = np.linalg.norm(np.array([rx_v, ry_v, rz_v]) - np.array([rx_p, ry_p, rz_p]))
        all_deg_error += euc_distance

        rot_mat_label, _ = cv2.Rodrigues(np.array([rx_v, ry_v, rz_v]))
        rot_mat_pred, _ = cv2.Rodrigues(np.array([rx_p, ry_p, rz_p]))

        theta = Utils.get_theta_between_rot_mats(rot_mat_pred, rot_mat_label)
        theta_error += theta

        # log.info('v_pose: %s\t\t%s\t\t%s' % (rx_v, ry_v, rz_v))
        # log.info('p_pose: %s\t\t%s\t\t%s' % (rx_p, ry_p, rz_p))
        log.info('theta: %s | euc. d.: %s' % (theta, euc_distance))

        total_samples_compared += 1

    log.info('all_deg_error:: %s' % (all_deg_error / total_samples_compared))

    return theta_error / total_samples_compared


def predict(data: [Data], model_input=None, limit=-1, is_6pos=False, model_name='c2_net'):
    if limit > -1:
        data = data[:limit]

    if data[0].bbox is None:
        data = DetectFace.get_face_bb_multithread(data)

    orig_data_len = len(data)
    # add extra pedding to match batches
    if len(data) % Model.BATCH_SIZE != 0:
        data += data[:(Model.BATCH_SIZE - len(data) % Model.BATCH_SIZE)]

    predicted_poses = Model.predict(data, model_input=model_input, model_name=model_name)

    predicted_poses = predicted_poses[:orig_data_len]
    data = data[:orig_data_len]

    avg_theta_error = validate_predictions(predicted_poses, data, is_6pos)
    log.info('AVG error: %s' % avg_theta_error)

    export_results(predicted_poses, data)

    return predicted_poses


def run_validation_set2(model_input=None, limit=-1, is_6pos=False, model_name='c2_net'):
    log.info('run_validation_set2::')
    data: [Data] = DataSources.load_validation_dataset2()
    predict(data, model_input, limit, is_6pos, model_name=model_name)


def test_300w_3d_helen1(model_input=None, limit=-1, is_6pos=False, model_name='c2_net'):
    log.info('run_validation_set2::')
    data: [Data] = DataSources.load_naive_augmented_dataset(DataSources.DataSources._300W_3D_HELEN_NG1, limit=limit)
    predict(data, model_input, limit, is_6pos, model_name=model_name)


def run_test_set(model_input=None, limit=-1, is_6pos=False, model_name='c2_net'):
    import GenerateTrainingSet
    log.info('run_validation_set2::')
    data: [Data] = DataSources.load_test_set()

    # find bbox for each image
    data = DetectFace.detect_face_multithread(data)

    # find landmarks
    data = DetectFace.detect_face_multithread(data, fn=DetectFace.detect_face_landmarks_dlib)

    # updated bboxes according to landmarks if such were found
    for data_ in data:
        if data_.landmarks_2d is not None:
            DetectFace.get_face_bb(data_)

    # estimate pose based on the dlib landmarks and the face model
    face_model = GenerateTrainingSet.get_face_model()
    for data_ in data:
        try:
            proj_mat = GenerateTrainingSet.calc_projection(data_.landmarks_2d, face_model.model_TD, face_model)
            pose = GenerateTrainingSet.estimate_pose_from_landmarks(proj_mat, face_model)
            data_.pose = pose
        except:
            log.warning('failed to get pose for image %s' % data_.image)

    predicted_poses = predict(data, model_input, limit, is_6pos, model_name=model_name)




def export_results(predicted_poses, data):
    with open('results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['file name', 'rx', 'ry', 'rz'])
        writer.writeheader()
        for i, predicted_pose in enumerate(predicted_poses):
            rx, ry, rz = predicted_pose
            writer.writerow({'file name' : data[i].image, 'rx': rx, 'ry': ry, 'rz': rz})


if __name__ == '__main__':
    # run_validation_set2(None)
    # run_validation_set2(model_input='models/transfer_3params/custom2_full_ds.ckpt', model_name='c2_net')
    # run_validation_set2(model_input='models/transfer_3params/c_resnet_full_ds.ckpt', model_name='c_resnet')
    # test_300w_3d_helen1(model_input='models/transfer_6params/cp_300w_3d_helen_naive_1.ckpt', limit=16, is_6pos=True)
    # test_300w_3d_helen1(model_input='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt', limit=150)
    # test_300w_3d_helen1(None, limit=500)
    run_test_set(model_input='models/transfer_3params/custom2_full_ds.ckpt', model_name='c2_net')
    # run_test_set(model_input='models/transfer_3params/custom2_full_ds_b2.ckpt', model_name='c2_net')
    # run_test_set(model_input='models/transfer_3params/c_resnet_full_ds.ckpt', model_name='c_resnet')
    # run_test_set(model_input='models/transfer_3params/c_resnet_full_ds_b2.ckpt', model_name='c_resnet')
