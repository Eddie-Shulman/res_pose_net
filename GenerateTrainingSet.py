import logging
import os

import cv2
import numpy as np

import DataSources
from Utils import Data
from detect_face import DetectFace
from lib.face_specific_augm import myutil, config, renderer
from lib.face_specific_augm.ThreeD_Model import FaceModel
import matplotlib.pyplot as plt


DEBUG = False

log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)


def naive_augment_300w_3d_helen():
    model_3d = '../datasets/model3D_aug_-00_00_01.mat'
    output_folder = '../augmented/300w_3d_helen_naive_v2'

    face_model = FaceModel(model_3d, 'model3D', False)
    limit = -1 if not DEBUG else 5
    data: [Data] = DataSources._load_data(DataSources.DataSources._300W_3D_HELEN,
                                                                   DataSources._300w_3d_parser, limit=limit)

    # save augmentation poses for training
    for i, data_elm in enumerate(data):
        if DEBUG and i > 5:
            break

        log.info('augment_validation_set:: image: %s | %s/%s' % (data_elm.image, i, len(data)))

        if DEBUG:
            show_landmarks_on_image(data_elm)

        generate_naive_augmentations(i, data_elm, output_folder, face_model)


def generate_naive_augmentations(img_index, data: Data, output_folder, face_model):
    x, y, w, h = DetectFace.get_face_bb2(data.landmarks_2d)
    face_center = (x + w)/2, (y + h)/2

    filename = os.path.basename(data.image)

    orig_img = cv2.imread(data.image, 1)

    orig_img_clone = np.array(orig_img)
    if DEBUG:
        cv2.rectangle(orig_img_clone, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.imwrite('%s/%s' % (output_folder, data.image), orig_img_clone)

    augmentation_transformations = generate_naive_augmentation_transformations(img_index % 4, orig_img.shape[1], face_center[0], face_center[1])

    transformation_data = []
    filename_, file_extension_ = os.path.splitext(data.image)
    filename_ = os.path.splitext(os.path.basename(data.image))[0]

    for i, augmentation_transformation in enumerate(augmentation_transformations):

        if DEBUG:
            log.info('generate_naive_augmentations:: image: %s | trans index: %s/%s' % (data.image, i, len(augmentation_transformations)))

        rotation_transformation, roll, flipped_rotation_transformation = augmentation_transformation

        for j, affine_transformation_matrix in enumerate([rotation_transformation, flipped_rotation_transformation]):
            transformed_image = cv2.warpAffine(orig_img, affine_transformation_matrix, (int(orig_img.shape[1]*1.), int(orig_img.shape[0]*1.)))

            tmp_landmarks = np.hstack((data.landmarks_2d, np.ones((data.landmarks_2d.shape[0], 1))))
            transformed_landmarks_2d = np.asmatrix(affine_transformation_matrix) * tmp_landmarks.transpose()
            transformed_landmarks_2d = transformed_landmarks_2d.transpose()

            proj_mat_t_land = calc_projection(transformed_landmarks_2d, face_model.model_TD, face_model)
            transformed_pose = estimate_pose_from_landmarks(proj_mat_t_land, face_model)

            if DEBUG:
                proj_mat_err = eval_projection_matrix(proj_mat_t_land, transformed_landmarks_2d, face_model)
                log.info('proj_mat_land_err: %s' % proj_mat_err)

                proj_mat_t_pose = calc_projection_via_pose(transformed_pose, face_model)
                proj_mat_err = eval_projection_matrix(proj_mat_t_pose, transformed_landmarks_2d, face_model)
                log.info('proj_mat_pose_err: %s' % proj_mat_err)

            if DEBUG:
                transformed_face_bb = DetectFace.get_face_bb2(transformed_landmarks_2d)
                t_x, t_y, t_w, t_h = transformed_face_bb
                cv2.rectangle(transformed_image, (int(t_x), int(t_y)), (int(t_x+t_w), int(t_y+t_h)), (0, 255, 0), 2)
                for transformed_landmark_2d in transformed_landmarks_2d:
                    cx, cy = np.array(transformed_landmark_2d, dtype=np.int)[0]
                    cv2.circle(transformed_image, (cx, cy), 2, (0, 255, 0), 2)

            transformation_data.append(['%s_%s_aug%s' % (filename_, i, file_extension_), transformed_pose, transformed_landmarks_2d])
            cv2.imwrite('%s/%s' % (output_folder, '%s--__%s_%s_aug%s' % (filename_, i, j, file_extension_)), transformed_image)

    with open('%s/%s' % (output_folder, '%s.meta' % filename_), 'w') as f:
        for transformation_data_ in transformation_data:
            f.write('%s|%s|%s\n' % (transformation_data_[0], np.array2string(transformation_data_[1]), np.array2string(transformation_data_[2]).replace('\n', ' ')))


def estimate_pose_from_landmarks(proj_mat_landmarks_2d, face_model):
    RT = np.linalg.inv(face_model.out_A) @ proj_mat_landmarks_2d
    translation_vec = np.array(RT.T[3])[0]
    R = np.array(RT.T[:3].T)
    rotation_vec, jacobian = cv2.Rodrigues(R, None)
    rotation_vec = rotation_vec.transpose()[0]
    transformed_pose = np.concatenate((rotation_vec, translation_vec))

    return transformed_pose


def calc_projection(landmarks_2d, landmarks_3d, face_model=None):
    if face_model is None:
        face_model = FaceModel('../datasets/model3D_aug_-00_00_01.mat', 'model3D', False)

    # in case there is a missing dimension
    if landmarks_3d.shape[1] == 2:
        landmarks_3d = np.hstack((landmarks_3d, np.ones([68, 1])))

    padded_landmarks = np.reshape(landmarks_3d, [68, 3, 1])
    landmarks_2d = np.reshape(landmarks_2d, [68, 2, 1])

    retval, rotation_vecs, translation_vecs = cv2.solvePnP(padded_landmarks, landmarks_2d, face_model.out_A, None)

    rotation_mat, jacobian = cv2.Rodrigues(rotation_vecs, None)

    RT = np.hstack((rotation_mat, np.reshape(translation_vecs, [3,1])))
    projection_matrix = face_model.out_A @ RT

    return projection_matrix


def calc_projection_via_pose(pose_params, face_model):
    rotation_vecs = np.array(pose_params[0:3])
    translation_vecs = np.array(pose_params[3:6])

    rotation_mat, jacobian = cv2.Rodrigues(rotation_vecs, None)

    RT = np.hstack((rotation_mat, np.reshape(translation_vecs, [3,1])))
    projection_matrix = face_model.out_A @ RT

    return projection_matrix


def eval_projection_matrix(proj_mat, landmarks_2d, face_model, landmarks_3d=None):
    if landmarks_3d is None:
        landmarks_3d = np.hstack((face_model.model_TD, np.ones([68, 1])))
    # else:
    #     landmarks_3d = np.hstack((landmarks_3d, np.ones([68, 1])))

    total_distance = 0

    for i, landmark_3d in enumerate(landmarks_3d):
        landmarks_2d_expacted = proj_mat * np.reshape(landmark_3d, [4,1])

        # normalize z=1
        landmarks_2d_expacted /= landmarks_2d_expacted[2]
        # make it readable
        landmarks_2d_expacted = np.squeeze(np.asarray(landmarks_2d_expacted))

        distance = np.linalg.norm(landmarks_2d_expacted[:2] - landmarks_2d[i])
        total_distance += distance

        # log.info('%s %s %s' % (landmarks_2d[i], landmarks_2d_expacted, distance))

    # log.info('avg. distance %s' % (total_distance/68))
    return total_distance/68


def get_translation_matrix(tx=0., ty=0., tz=0.):
    translation_matrix = np.vstack((np.identity(3, dtype=np.float32), np.array([0, 0, 0])))
    translation_matrix = np.hstack((translation_matrix, np.reshape(np.array([tx, ty, tz, 1]), [4,1])))

    return translation_matrix


def get_scale_matrix(scale=1.):
    scale_matrix = scale * np.identity(3)
    scale_matrix = np.vstack((scale_matrix, np.array([0,0,0])))
    scale_matrix = np.hstack((scale_matrix, np.reshape(np.array([0,0,0,1]), [4,1])))

    return scale_matrix


def get_rotation_matrix(d_rx, d_ry, d_rz):
    rotation_matrix, jac = cv2.Rodrigues(np.array([np.deg2rad(d_rx), np.deg2rad(d_ry), np.deg2rad(d_rz)]).astype(np.float32), None)
    rotation_matrix = np.vstack((rotation_matrix, np.array([0,0,0])))
    rotation_matrix = np.hstack((rotation_matrix, np.reshape(np.array([0,0,0,1]), [4,1])))

    return rotation_matrix


def get_transformation_mat(d_rx=0., d_ry=0., d_rz=0., d_tx=0., d_ty=0., d_scale=1.):
    rotation_matrix = get_rotation_matrix(d_rx, d_ry, d_rz)
    translation_matrix = get_translation_matrix(d_tx, d_ty, 0)
    scale_matrix = get_scale_matrix(d_scale)

    transformation_mat = rotation_matrix @ translation_matrix @ scale_matrix
    return transformation_mat


def generate_augmentation_transformations(pitch=0, yaw=0, roll=0):
    augmentation_transformations = []
    augmentation_transformations_poses = []
    pitch, yaw, roll = int(np.rad2deg(pitch)), int(np.rad2deg(yaw)), int(np.rad2deg(roll))
    for d_rx in range(pitch-1, pitch+1, 1):  # up - down
        for d_ry in range(yaw-4, yaw+4, 2):  # left - right
            for d_rz in range(roll-1, roll+1, 1):  # circle
                for scale in range(100, 151, 25):
                    transformation_mat = get_transformation_mat(d_rx, d_ry, d_rz, d_scale=scale/100.)
                    augmentation_transformations.append(transformation_mat)
                    augmentation_transformations_poses.append([d_rx, d_ry, d_rz, 0., 0., scale/100.])

    return augmentation_transformations, augmentation_transformations_poses


def generate_naive_augmentation_transformations(augmentation_group_index, img_w, c_x, c_y):
    augmentation_transformations = []

    # generate rotations
    if augmentation_group_index == 0:
        for roll in [-11, 0, 5]:
                rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), roll, 1.)
                augmentation_transformations.append([rotation_matrix, np.deg2rad(roll)])
    elif augmentation_group_index == 1:
        for roll in [-7, 0, 13]:
                rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), roll, 1.)
                augmentation_transformations.append([rotation_matrix, np.deg2rad(roll)])
    elif augmentation_group_index == 2:
        for roll in [-19, 0, 31]:
                rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), roll, 1.)
                augmentation_transformations.append([rotation_matrix, np.deg2rad(roll)])
    else:
        for roll in [-29, 0, 17]:
                rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), roll, 1.)
                augmentation_transformations.append([rotation_matrix, np.deg2rad(roll)])

    # add flips to rotations
    flip_v_m = np.float32([[-1, 0, img_w - 1], [0, 1, 0]])
    for augmentation_transformation in augmentation_transformations:

        flip_v_m_ = np.identity(4)
        flip_v_m_[:2,:3] = flip_v_m

        augmentation_transformation_ = np.identity(4)
        augmentation_transformation_[:2, :3] = augmentation_transformation[0]

        flip_transformation = flip_v_m_ @ augmentation_transformation_
        augmentation_transformation.append(flip_transformation[:2, :3])

    return augmentation_transformations


def show_landmarks_on_image(data: Data):
    tmp_image = cv2.imread(data.image)
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    for landmark_2d in data.landmarks_2d:
        landmark_2d_ = landmark_2d.astype(np.int)
        x, y = landmark_2d_
        cv2.circle(tmp_image, (x, y), 2, (255, 0, 0), 2)

    data = DetectFace.get_face_bb(data)
    x, y, w, h = data.bbox
    cv2.rectangle(tmp_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    plt.figure()
    plt.imshow(tmp_image, cmap='gray', interpolation='nearest')
    plt.show()


if __name__ =='__main__':
    # augment_300w_lp_helen()
    # augment_validation_set()
    # naive_augment_validation_set()
    naive_augment_300w_3d_helen()
