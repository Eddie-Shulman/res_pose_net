import logging

import DataSources
import Model
from detect_face import DetectFace


log = logging.getLogger('DataSources')


def train_300w_3d_helen_naive_augmentations(fast_face_detect=True, limit=-1):
    log.info('train_300w_3d_helen_naive_augmentations::')
    images_v, validation_pose_v, landmarks_2d_v = DataSources.load_validation_dataset2()
    # images, validation_pose, landmarks_2d = DataSources._load_data(DataSources.DataSources._300W_3D_HELEN_NG1, DataSources._300w_3d_parser, limit=limit)
    images, validation_pose, landmarks_2d = DataSources.load_naive_augmented_dataset(DataSources.DataSources._300W_3D_HELEN_NG1, limit=limit)

    if limit > -1:
        images, validation_pose, landmarks_2d = images[:limit], validation_pose[:limit], landmarks_2d[:limit]

    log.info('train_300w_3d_helen_naive_augmentations:: start face bbox detections')
    if fast_face_detect:
        b_boxes = DetectFace.get_face_bb_multithread(landmarks_2d)
        b_boxes_v = DetectFace.get_face_bb_multithread(landmarks_2d_v)
    else:
        b_boxes = DetectFace.detect_face_multithread(images)
        b_boxes_v = DetectFace.detect_face_multithread(images_v)

    none_indices = [i for i, x in enumerate(b_boxes) if x is None]

    images = [image for i, image in enumerate(images) if i not in none_indices]
    validation_pose = [validation_pose_ for i, validation_pose_ in enumerate(validation_pose) if i not in none_indices]
    b_boxes = [b_box for i, b_box in enumerate(b_boxes) if i not in none_indices]

    Model.train(images, b_boxes, validation_pose, images_v, b_boxes_v, validation_pose_v , 15,
                    model_input=None,
                    model_output='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt')


if __name__ == '__main__':
    train_300w_3d_helen_naive_augmentations()
