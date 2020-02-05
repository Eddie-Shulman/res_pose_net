import logging

import cv2

import DataSources
import Model
from detect_face import DetectFace
import Utils

log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)


def train_300w_3d_helen_naive_augmentations(data_sources: [DataSources.DataSources],
                                            model_input, model_output,
                                            fast_face_detect=True, limit=-1):
    log.info('train_300w_3d_helen_naive_augmentations::')
    images_v, validation_pose_v, landmarks_2d_v = DataSources.load_validation_dataset2()

    images, validation_pose, landmarks_2d = [], [], []
    for data_source in data_sources:
        images_, validation_pose_, landmarks_2d_ = DataSources.load_naive_augmented_dataset(data_source, limit=limit)
        images += images_
        validation_pose += validation_pose_
        landmarks_2d += landmarks_2d_

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

    # for i, image in enumerate(images):
    #     log.info(image)
    #
    #     Utils.matplot_image(Model.load_image(image, b_boxes[i]))
    #
    #     image_ = cv2.imread(image)
    #     image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    #     image_ = Utils.pre_process_image2(image_, b_boxes[i])
    #     Utils.matplot_image(image_)
    #
    #     image = cv2.imread(image)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = Utils.pre_process_image(image, b_boxes[i])
    #     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    #     Utils.matplot_image(image)

    Model.train(images, b_boxes, validation_pose, images_v, b_boxes_v, validation_pose_v ,
                epochs=15,
                model_input=model_input,
                model_output=model_output)


if __name__ == '__main__':
    # train_300w_3d_helen_naive_augmentations([DataSources.DataSources._300W_3D_HELEN_NG1],
    #                                         model_input=None,
    #                                         model_output='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt',
    #                                         limit=-1)
    train_300w_3d_helen_naive_augmentations([DataSources.DataSources._300W_3D_HELEN_NG1, DataSources.DataSources._300W_3D_HELEN_NG2, DataSources.DataSources._300W_3D_HELEN_NG3, DataSources.DataSources._300W_3D_HELEN_NG4],
                                            model_input=None,
                                            model_output='models/transfer_3params/cp_300w_3d_helen_naive.ckpt',
                                            limit=-1)
    # train_300w_3d_helen_naive_augmentations(DataSources.DataSources._300W_3D_HELEN_NG2,
    #                                         model_input='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt',
    #                                         model_output='models/transfer_3params/cp_300w_3d_helen_naive_2.ckpt',
    #                                         limit=1000)
