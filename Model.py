import logging
import math
import os
import traceback

import cv2
import numpy as np
import tensorflow as tf
import time

import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications import InceptionResNetV2, ResNet50
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling3D, AveragePooling2D, \
    Flatten
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


import Utils

BATCH_SIZE = 16
RESNET_SIZE = 224
USE_6POSE = False
USE_ADAM_OPT = False


log = logging.getLogger('DataSources')
log.setLevel(logging.DEBUG)


def load_image(image: str, bbox: np.array=None):
    image_array = cv2.imread(image)  # BGR
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    if bbox is not None:
        try:
            image_array = Utils.pre_process_image2(image_array, bbox)
        except:
            log.exception('failed to pre process image %s with bbox %s' % (image, bbox))
            raise Exception('failed to pre process image %s with bbox %s' % (image, bbox))
    image_array = cv2.resize(image_array, (RESNET_SIZE, RESNET_SIZE), interpolation=cv2.INTER_CUBIC)
    image_array = np.asarray(image_array)
    return image_array


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images, labels, b_boxes, batch_size, shuffle=False) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.b_boxes = b_boxes
        self.batch_size = batch_size
        self.shuffle = shuffle

        seq_len = int(np.floor(len(self.images) / self.batch_size))

        self.indexes = np.arange(start=0, stop=seq_len, step=1, dtype=np.uint16) * self.batch_size
        # initial data shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        index = self.indexes[idx]

        images = []
        labels = []
        for i in range(index, index + self.batch_size):
            bbox = self.b_boxes[i] if self.b_boxes is not None and len(self.b_boxes) > 0 else None
            image_array = load_image(self.images[i], bbox)
            images.append(image_array)
            label = self.labels[i] if USE_6POSE is True else self.labels[i][:3]
            label = np.array(label, dtype=np.float32)
            labels.append(label)

        return np.array(images), np.array(labels, dtype=np.float32)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.shuffle:
            np.random.shuffle(self.indexes)


class PredictDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images, b_boxes, batch_size) -> None:
        super().__init__()
        self.images = images
        self.b_boxes = b_boxes
        self.batch_size = batch_size

        seq_len = int(np.floor(len(self.images) / self.batch_size))

        self.indexes = np.arange(start=0, stop=seq_len, step=1, dtype=np.uint16) * self.batch_size

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.batch_size):
            bbox = self.b_boxes[i] if self.b_boxes is not None and len(self.b_boxes) > 0 else None
            image_array = load_image(self.images[i], bbox)
            images.append(image_array)

        return np.array(images)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))


def custom_acc(y_true, y_pred):
    mean_theta = custom_loss(y_true, y_pred)
    correct_prediction = tf.constant(180., tf.float32) - (mean_theta * 180. / np.pi)
    return correct_prediction


def custom_loss(y_true, y_pred):
    theta_errors = []

    for i in range(0, BATCH_SIZE):
        rot_vec_l = y_true[i][:3]
        rot_vec_p = y_pred[i][:3]

        R_l = Utils.rodrigues_batch(tf.reshape(rot_vec_l, [-1, 3]))
        R_l = tf.reshape(R_l, (3, 3))
        R_p = Utils.rodrigues_batch(tf.reshape(rot_vec_p, [-1, 3]))
        R_p = tf.reshape(R_p, (3, 3))

        # calc angle between rotation matrix
        theta_error = tf.math.acos((tf.linalg.trace(tf.matmul(tf.transpose(R_p), R_l)) - 1) / 2)

        theta_error = tf.math.abs(theta_error)
        theta_errors.append(theta_error)

    return tf.reduce_mean(tf.convert_to_tensor(theta_errors))
    # return tf.reduce_mean(y_pred)


def custom_loss_2(y_true, y_pred):

    angles_true = tf.slice(y_true, [0, 0], [BATCH_SIZE,3])
    angles_pred = tf.slice(y_pred, [0, 0], [BATCH_SIZE,3])

    translations_true = tf.slice(y_true, [0, 3], [BATCH_SIZE,3])
    translations_pred = tf.slice(y_pred, [0, 3], [BATCH_SIZE,3])

    return tf.reduce_mean(tf.compat.v1.math.square(angles_true - angles_pred)) + tf.math.sqrt(tf.compat.v1.math.reduce_euclidean_norm(translations_true - translations_pred))


def custom_loss_3(y_true, y_pred):

    rx_l = tf.slice(y_true, [0, 0], [BATCH_SIZE,1])
    rx_p = tf.slice(y_pred, [0, 0], [BATCH_SIZE,1])

    ry_l = tf.slice(y_true, [0, 1], [BATCH_SIZE,1])
    ry_p = tf.slice(y_pred, [0, 1], [BATCH_SIZE,1])

    rz_l = tf.slice(y_true, [0, 2], [BATCH_SIZE,1])
    rz_p = tf.slice(y_pred, [0, 2], [BATCH_SIZE,1])

    return tf.reduce_mean(tf.compat.v1.math.square(rx_l - rx_p)) * 0.5 + \
           tf.reduce_mean(tf.compat.v1.math.square(ry_l - ry_p)) * 0.3 + \
           tf.reduce_mean(tf.compat.v1.math.square(rz_l - rz_p)) * 0.2


def get_model(train_mode=True):

    input = Input(shape=(RESNET_SIZE, RESNET_SIZE, 3))
    # base = ResNet50(input_tensor=input, include_top=False, weights='imagenet', pooling='max')
    base = ResNet50(input_tensor=input, include_top=False, weights='imagenet')
    # base = InceptionResNetV2(input_tensor=input,
    #                          include_top=False,
    #                          weights='imagenet')

    base.trainable = False
    # for layer in base.layers:
    #     layer.trainable = False

    x = Dense(1024, activation='relu', name='pose_dense_1')(base.output)
    x = Flatten(name='pose_flatten_2')(x)
    x = Dense(512, activation='relu', name='pose_dense_3')(x)
    x = Dense(256, activation='relu', name='pose_dense_4')(x)
    x = Dense(128, activation='relu', name='pose_dense_5')(x)
    x = Dense(64, activation='relu', name='pose_dense_6')(x)

    if USE_6POSE is True:
        out = Dense(6, activation='softmax', name='pose_dense_ouptut')(x)
    else:
        out = Dense(3, activation='tanh', name='pose_dense_ouptut')(x)

    model = Model(inputs=base.inputs, outputs=out)

    if USE_ADAM_OPT is True:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    else:
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    if train_mode:
        model.compile(optimizer, loss=custom_loss_3, metrics=['accuracy'])  # mse -> mean sqare error | 'accuracy'
    else:
        model.compile(optimizer, loss='mse', metrics=['accuracy'])  # mse -> mean sqare error | 'accuracy'

    model.summary()

    return model


def train(images, b_boxes, labels, images_v, b_boxes_v, labels_v, epochs=1, model_input=None, model_output=None):

    train_data_generator = DataGenerator(images, labels, b_boxes, BATCH_SIZE, shuffle=True)
    valid_data_generator = DataGenerator(images_v, labels_v, b_boxes_v, BATCH_SIZE)

    model = get_model()

    callbacks = []

    if model_input is not None:
        # Loads the weights
        log.info('train loading weights')
        model.load_weights(model_input)

    if model_output is not None:
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_output,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)

    history = model.fit_generator(train_data_generator,
                                  epochs=epochs, verbose=1,
                                  validation_data=valid_data_generator,
                                  callbacks=callbacks
                                  )

    eval_data_generator = DataGenerator(images[:500], labels[:500], b_boxes[:500], BATCH_SIZE)
    eval = model.evaluate_generator(eval_data_generator)

    log.info('train eval: %s' % eval)

    log.info('\nhistory dict:', history.history)

    bbox = b_boxes[0] if b_boxes is not None and len(b_boxes) > 0 else None
    image_array = load_image(images[0], bbox)

    prediction = model.predict(np.array([image_array]))
    log.info(prediction)
    log.info(labels[0][:3])

    R_p, _ = cv2.Rodrigues(prediction[0])
    R_l, _ = cv2.Rodrigues(labels[0][:3])
    theta = np.arccos((np.trace(R_p.T @ R_l) - 1) / 2)
    log.info(np.rad2deg(theta))


def predict(images, b_boxes, labels, model_input):
    model = get_model(train_mode=False)

    if model_input is not None:
        model.load_weights(model_input)

    # train_data_generator = DataGenerator(images[:1], [[0.110099974, 0.238761767, -0.443073971 ,40.78644871 ,34.50543871 ,1035.096866]], b_boxes[:1], 1)
    # model.fit_generator(train_data_generator)

    prediction_data_generator = PredictDataGenerator(images, b_boxes, BATCH_SIZE)
    data_generator = DataGenerator(images, labels, b_boxes, BATCH_SIZE)
    predictions = model.predict_generator(prediction_data_generator, verbose=1)

    eval_loss = model.evaluate_generator(data_generator)
    log.info('predict eval loss: %s' % eval_loss)

    return predictions
