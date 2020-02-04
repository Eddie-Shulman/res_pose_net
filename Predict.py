import logging

import DataSources
from detect_face import DetectFace
import Model
import numpy as np
import cv2

log = logging.getLogger('DataSources')


def validate_predictions(predicted_poses, validation_poses):

    theta_error = 0.
    all_deg_error = 0

    for i, predicted_pose in enumerate(predicted_poses):
        rx_p, ry_p, rz_p = predicted_pose
        rx_v, ry_v, rz_v, tx_l, ty_l, tz_l = validation_poses[i]

        all_deg_error += np.linalg.norm(np.array([rx_v, ry_v, rz_v]) - np.array([rx_p, ry_p, rz_p]))

        rot_mat_label, _ = cv2.Rodrigues(np.array([rx_v, ry_v, rz_v]))
        rot_mat_pred, _ = cv2.Rodrigues(np.array([rx_p, ry_p, rz_p]))

        theta = np.arccos((np.trace(rot_mat_pred.T @ rot_mat_label) - 1) / 2)
        theta_error += np.rad2deg(theta)

    print('all_deg_error:: %s' % (all_deg_error / len(predicted_poses)))

    return theta_error / len(predicted_poses)


def predict(images, pose, landmarks_2d, limit=-1):
    if limit > -1:
        images, pose = images[:limit], pose[:limit]

    b_boxes = DetectFace.get_face_bb_multithread(landmarks_2d)

    predicted_poses = Model.predict(images, b_boxes, model_input='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt')

    avg_theta_error = validate_predictions(predicted_poses, pose)
    print('AVG error: %s' % avg_theta_error)


def run_validation_set2(limit=-1):
    print('run_validation_set2::')
    images, pose, landmarks_2d = DataSources.load_validation_dataset2()
    predict(images, pose, landmarks_2d, limit)


def test_300w_3d_helen1(limit=-1):
    print('run_validation_set2::')
    images, pose, landmarks_2d = DataSources.load_naive_augmented_dataset(DataSources.DataSources._300W_3D_HELEN_NG1)
    predict(images, pose, landmarks_2d, limit)


if __name__ == '__main__':
    run_validation_set2()
    # test_300w_3d_helen1(1000)
