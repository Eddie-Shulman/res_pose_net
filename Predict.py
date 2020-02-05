import logging

import DataSources
from detect_face import DetectFace
import Model
import numpy as np
import cv2

from Settings import *

log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)
# log.addHandler(ch)


def validate_predictions(predicted_poses, validation_poses, is_6pos=False):

    theta_error = 0.
    all_deg_error = 0

    for i, predicted_pose in enumerate(predicted_poses):
        if is_6pos:
            rx_p, ry_p, rz_p, tx_p, ty_p, tz_p = predicted_pose
        else:
            rx_p, ry_p, rz_p = predicted_pose

        rx_v, ry_v, rz_v, tx_l, ty_l, tz_l = validation_poses[i]

        euc_distance = np.linalg.norm(np.array([rx_v, ry_v, rz_v]) - np.array([rx_p, ry_p, rz_p]))
        all_deg_error += euc_distance

        rot_mat_label, _ = cv2.Rodrigues(np.array([rx_v, ry_v, rz_v]))
        rot_mat_pred, _ = cv2.Rodrigues(np.array([rx_p, ry_p, rz_p]))

        theta = np.arccos((np.trace(rot_mat_pred.T @ rot_mat_label) - 1) / 2)
        theta = np.rad2deg(np.abs(theta))
        theta_error += theta

        log.info('v_pose: %s\t%s\t%s' % (rx_v, ry_v, rz_v))
        log.info('p_pose: %s\t%s\t%s' % (rx_p, ry_p, rz_p))
        log.info('theta: %s | euc. d.: %s' % (theta, euc_distance))

    log.info('all_deg_error:: %s' % (all_deg_error / len(predicted_poses)))

    return theta_error / len(predicted_poses)


def predict(images, pose, landmarks_2d, model_input=None, limit=-1, is_6pos=False):
    if limit > -1:
        images, pose = images[:limit], pose[:limit]

    b_boxes = DetectFace.get_face_bb_multithread(landmarks_2d)

    predicted_poses = Model.predict(images, b_boxes, pose, model_input=model_input)

    avg_theta_error = validate_predictions(predicted_poses, pose, is_6pos)
    log.info('AVG error: %s' % avg_theta_error)


def run_validation_set2(model_input=None, limit=-1, is_6pos=False):
    log.info('run_validation_set2::')
    images, pose, landmarks_2d = DataSources.load_validation_dataset2()
    predict(images, pose, landmarks_2d, model_input, limit, is_6pos)


def test_300w_3d_helen1(model_input=None, limit=-1, is_6pos=False):
    log.info('run_validation_set2::')
    images, pose, landmarks_2d = DataSources.load_naive_augmented_dataset(DataSources.DataSources._300W_3D_HELEN_NG1, limit=limit)
    predict(images[500:1000], pose[500:1500], landmarks_2d[500:1500], model_input, limit, is_6pos)


if __name__ == '__main__':
    # run_validation_set2(None)
    # run_validation_set2(model_input='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt')
    # test_300w_3d_helen1(model_input='models/transfer_6params/cp_300w_3d_helen_naive_1.ckpt', limit=16, is_6pos=True)
    test_300w_3d_helen1(model_input='models/transfer_3params/cp_300w_3d_helen_naive.ckpt', limit=2000)
    # test_300w_3d_helen1(None, limit=500)
