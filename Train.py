import logging

import cv2

import DataSources
import Model
from detect_face import DetectFace
import Utils
from Utils import Data

import numpy as np

log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)


def run_train(data: [Data], data_v: [Data], fast_face_detect=True, epochs=15, model_input=None, model_output=None):

    log.info('train_300w_3d_helen_naive_augmentations:: start face bbox detections')
    if fast_face_detect:
        data: [Data] = DetectFace.get_face_bb_multithread(data)
        data2 = DetectFace.get_face_bboxes(data)

        for i, data_elm in enumerate(data):
            x1, y1, w1, h1 = data_elm.bbox
            x2, y2, w2, h2 = data2[i].bbox

            if x1 != x2 or y1 != y2 or w1 != w2 or h1 != h2:
                log.error('INVALID BBOXES !!!')
                raise Exception('INVALID BBOXES !!!')

        data_v = DetectFace.get_face_bb_multithread(data_v)
    else:
        # TODO: refactor code to use data obj
        b_boxes = DetectFace.detect_face_multithread(data)
        b_boxes_v = DetectFace.detect_face_multithread(data_v)

    # none_indices = [i for i, x in enumerate(b_boxes) if x is None]
    #
    # images = [image for i, image in enumerate(images) if i not in none_indices]
    # validation_pose = [validation_pose_ for i, validation_pose_ in enumerate(validation_pose) if i not in none_indices]
    # b_boxes = [b_box for i, b_box in enumerate(b_boxes) if i not in none_indices]

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

    Model.train(data, data_v , epochs=epochs, model_input=model_input, model_output=model_output)


def train_300w_3d_helen_naive_augmentations(data_sources: [DataSources.DataSources],
                                            model_input, model_output,
                                            limit=-1):
    log.info('train_300w_3d_helen_naive_augmentations::')
    data_v: [Data] = DataSources.load_validation_dataset2(recalc_pose=True)

    data: [Data] = []

    for data_source in data_sources:
        data += DataSources.load_naive_augmented_dataset(data_source, limit=limit)

    if limit > -1:
        np.random.shuffle(data)
        data = data[:limit]

    run_train(data, data_v, model_input=model_input, model_output=model_output, epochs=30)


def train_validation_set_2(model_input=None, model_output=None):

    data_v = DataSources.load_validation_dataset2()
    data = DataSources.load_naive_augmented_dataset(DataSources.DataSources.VALIDATION_SET2_NG)

    # data = data[:]
    run_train(data, data_v, model_input=model_input, model_output=model_output, epochs=50)


if __name__ == '__main__':
    # train_validation_set_2(model_output='models/validation_set2/cp_validation_set2_naive.ckpt')

    train_300w_3d_helen_naive_augmentations([DataSources.DataSources._300W_3D_HELEN_V2,
                                             DataSources.DataSources._300W_3D_LFPW_NG,
                                             DataSources.DataSources.AFLW2000_NG,
                                             DataSources.DataSources.VALIDATION_SET2_NG
                                             ],
                                            model_input='models/transfer_3params/c_resnet_full_ds.ckpt',
                                            model_output='models/transfer_3params/c_resnet_full_ds_b2.ckpt',
                                            limit=-1)
    # train_300w_3d_helen_naive_augmentations([DataSources.DataSources._300W_3D_HELEN_NG1, DataSources.DataSources._300W_3D_HELEN_NG2,
    #                                          DataSources.DataSources._300W_3D_HELEN_NG3, DataSources.DataSources._300W_3D_HELEN_NG4],
    #                                         model_input=None,
    #                                         model_output='models/transfer_3params/cp_300w_3d_helen_naive.ckpt',
    #                                         limit=-1)
    # train_300w_3d_helen_naive_augmentations(DataSources.DataSources._300W_3D_HELEN_NG2,
    #                                         model_input='models/transfer_3params/cp_300w_3d_helen_naive_1.ckpt',
    #                                         model_output='models/transfer_3params/cp_300w_3d_helen_naive_2.ckpt',
    #                                         limit=1000)
