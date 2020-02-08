import csv
import json
import logging
import os
import re
from enum import Enum
from functools import partial
from multiprocessing.pool import Pool
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from Utils import Data

log = logging.getLogger('DataSources')


class DataSources(Enum):

    _300W_3D_HELEN  = ('../datasets/300W-3D/HELEN', 'jpg')
    _300W_3D_LFPW   = ('../datasets/300W-3D/LFPW', 'jpg')
    AFLW2000        = ('../datasets/AFLW2000', 'jpg')

    _300W_3D_HELEN_V2   = ('../augmented/300w_3d_helen_naive_v2', 'jpg')
    _300W_3D_LFPW_NG    = ('../augmented/300w_3d_lfpw_naive', 'jpg')
    AFLW2000_NG         = ('../augmented/AFLW2000', 'jpg')
    VALIDATION_SET2_NG  = ('../augmented/validation_set2', 'png')

    _300W_3D_HELEN_NG1 = ('../augmented/300w_3d_helen_naive_1', 'jpg')
    _300W_3D_HELEN_NG2 = ('../augmented/300w_3d_helen_naive_2', 'jpg')
    _300W_3D_HELEN_NG3 = ('../augmented/300w_3d_helen_naive_3', 'jpg')
    _300W_3D_HELEN_NG4 = ('../augmented/300w_3d_helen_naive_4', 'jpg')

    VALIDATION_1 = ('../datasets/openu_valid_set1', 'png')
    VALIDATION_2 = ('../datasets/openu_valid_set2', 'png')

    TEST_SET = ('../datasets/openu_test_set', 'png')


def load_test_set(data_source: DataSources=DataSources.TEST_SET) -> [Data]:
    root_folder, image_type = data_source.value

    data: [Data] = []
    for root, dirs, files in os.walk("%s" % root_folder, topdown=False):
        for name in files:
            if name.endswith('.%s' % image_type):
                data.append(Data('%s/%s' % (root_folder, name), None, None))

    return data


def load_validation_set(data_source: DataSources=DataSources.VALIDATION_1) -> [Data]:
    root_folder, image_type = data_source.value

    data: [Data] = []
    with open('./%s/validation_set.csv' % root_folder, newline='') as csv_file:
        validation_csv = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in validation_csv:
            if row[0]:
                image = row[1]
                rx, ry, rz, tx, ty, tz = np.array(row[2:8]).astype(np.float)
                scale = tz
                image_ = './%s/images/%s' % (root_folder, image)
                pose = np.array([scale, rx, ry, rz, tx, ty])

                with open('./%s/images/%s' % (root_folder, image.replace('.png', '.pts'))) as pts_file:
                    rows = [rows.strip() for rows in pts_file]
                rows = rows[rows.index('{') + 1: rows.index('}')]
                landmarks_2d = np.array([row.split(' ') for row in rows]).astype(np.float)

                data.append(Data(image_, landmarks_2d, pose))

    return data


def load_validation_dataset2(data_source: DataSources=DataSources.VALIDATION_2, recalc_pose=True) -> [Data]:
    root_folder, image_type = data_source.value

    log.info('load_validation_data2::%s:: started' % data_source.name)
    data = []

    from GenerateTrainingSet import get_face_model, calc_projection, estimate_pose_from_landmarks
    face_model = get_face_model()

    with open('%s/valid_set2.csv' % root_folder, newline='') as csv_file:
        validation_csv = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in validation_csv:
            if row[0]:
                image = row[1]
                landmarks = np.array([np.array(re.sub('[\[\]]', '', re.sub('[ ]+', ' ', landmark)).strip().split(' '), dtype=np.float32) for landmark in row[8:]])
                if recalc_pose:
                    proj_mat_t_land = calc_projection(landmarks, face_model.model_TD, face_model)
                    pose = estimate_pose_from_landmarks(proj_mat_t_land, face_model)
                else:
                    pose = np.array(row[2:8], dtype=np.float32)
                data.append(Data('%s/%s' % (root_folder, image), landmarks, pose))

    return data


def load_naive_augmented_dataset(data_source: DataSources, limit=-1) -> [Data]:
    root_folder, image_type = data_source.value

    log.info('load_naive_augmented_validation_set::%s' % data_source.name)

    data: [Data] = []
    meta_data = {}

    for root, dirs, files in os.walk(root_folder, topdown=False):
        for name in files:
            if -1 < limit == len(data):
                break

            if name.endswith('_aug.%s' % image_type):

                if name not in  meta_data:

                    meta_file_name = name.split('--')
                    meta_file_name = '%s.meta' % meta_file_name[0]

                    with open('%s/%s' % (root_folder, meta_file_name)) as f:
                        meta_lines = f.readlines()

                    for meta_line in meta_lines:
                        file_name, pose, landmarks_2d_ = meta_line.split('|')
                        meta_data[file_name] = (np.array(json.loads(pose.replace(' ', ','))),
                                                np.array(json.loads(landmarks_2d_.replace('  ', ' ').replace(' ', ','))))

                rx, ry, rz, tx, ty, tz = meta_data[name][0]
                data.append(Data('%s/%s' % (root_folder, name),
                                 meta_data[name][1],
                                 np.array([rx, ry, rz, tx, ty, tz])))

    return data


def _300w_3d_parser(name: str, root_folder:str, image_type: str, landmarks_fld_name='pt2d') -> Data:
    image = '%s/%s' % (root_folder, name)
    meta = loadmat('%s/%s' % (root_folder, name.replace('.%s' % image_type, '.mat')))
    rx, ry, rz, tx, ty, tz, scale = meta["Pose_Para"].reshape([-1]).astype(np.float32)  # pitch yaw roll

    # convert to rotvec
    r = R.from_euler('zxy', [rx, ry, rz], degrees=False)
    rx, ry, rz = r.as_rotvec()

    pose = np.array([rx, ry, rz, tx, ty, tz, scale])

    landmarks_2d = (meta[landmarks_fld_name]).astype(np.float32).transpose()
    return Data(image, landmarks_2d, pose)


def _load_data(data_source: DataSources, parser_fn, limit=-1, threads=5, landmarks_fld_name='pt2d') -> [Data]:
    root_folder, image_type = data_source.value

    log.info('data_sources::%s:: started' % data_source.name)

    filenames = []

    for root, dirs, files in os.walk("%s" % root_folder, topdown=False):
        for name in files:
            if -1 < limit == len(filenames):
                break
            elif name.endswith('.%s' % image_type):
                filenames.append(name)

    batch_size = 500
    p = Pool(processes=threads)
    data: [Data] = []
    for i in range(int(np.floor(len(filenames) / batch_size))):
        log.info('data_sources::%s:: %s/%s' % (data_source.name, i*batch_size, len(filenames)))
        data += p.map(partial(parser_fn, root_folder=root_folder, image_type=image_type, landmarks_fld_name=landmarks_fld_name), filenames[i * batch_size: (i + 1) * batch_size])

    if len(filenames) % batch_size > 0:
        data += p.map(partial(parser_fn, root_folder=root_folder, image_type=image_type, landmarks_fld_name=landmarks_fld_name), filenames[int(np.floor(len(filenames) / batch_size) * batch_size):])

    p.close()
    p.join()

    return data
