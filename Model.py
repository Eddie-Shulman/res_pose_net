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
    Flatten, Dropout, Activation, GlobalMaxPooling2D
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


import Utils
from Utils import Data


BATCH_SIZE = 32
RESNET_SIZE = 224
USE_6POSE = False
USE_ADAM_OPT = True


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

    def __init__(self, data: [Utils.Data], batch_size, shuffle=False) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        seq_len = int(np.floor(len(self.data) / self.batch_size))

        self.indexes = np.arange(start=0, stop=seq_len, step=1, dtype=np.uint16) * self.batch_size
        # initial data shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        index = self.indexes[idx]

        images, poses = [], []

        for i in range(index, index + self.batch_size):
            data = self.data[i]
            bbox = data.bbox
            image_array = load_image(data.image, bbox)
            images.append(image_array)
            pose = data.pose if USE_6POSE is True else data.pose[:3]
            poses.append(np.array(pose, dtype=np.float32))

        return np.array(images), np.array(poses, dtype=np.float32)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.shuffle:
            np.random.shuffle(self.indexes)


class PredictDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size

        seq_len = int(np.floor(len(self.data) / self.batch_size))

        self.indexes = np.arange(start=0, stop=seq_len, step=1, dtype=np.uint16) * self.batch_size

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.batch_size):
            data = self.data[i]
            bbox = data.bbox
            image_array = load_image(data.image, bbox)
            images.append(image_array)

        return np.array(images)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))


def custom_acc(y_true, y_pred):
    # mean_theta = tf.reduce_mean(tf.math.reduce_euclidean_norm(y_true - y_pred, 1))
    mean_theta = custom_loss_3(y_true, y_pred)
    # mean_theta = custom_loss(y_true, y_pred)
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

    return tf.reduce_mean(tf.norm(y_true - y_pred))  # return euclidean

    # return tf.reduce_mean(tf.nn.l2_loss(y_true-y_pred))  # same as norm but without the sqrt
    # return tf.reduce_mean(tf.keras.layers.Lambda(lambda y_true, y_pred: tf.math.reduce_euclidean_norm(y_true - y_pred, 1)))

    # return (tf.reduce_mean(tf.compat.v1.math.abs(rx_l - rx_p)) * 1. +
    #        tf.reduce_mean(tf.compat.v1.math.abs(ry_l - ry_p)) * 1. +
    #        tf.reduce_mean(tf.compat.v1.math.abs(rz_l - rz_p)) * 1.) / 3


def get_model(train_mode=True):

    input = Input(shape=(RESNET_SIZE, RESNET_SIZE, 3))
    # base = ResNet50(input_tensor=input, include_top=False, weights='imagenet')
    base = ResNet50(input_tensor=input, include_top=False, weights='imagenet')
    # base = InceptionResNetV2(input_tensor=input,
    #                          include_top=False,
    #                          weights='imagenet')

    base.trainable = False
    # for layer in base.layers:
    #     layer.trainable = False

    x = base.output

    # op 2
    # x = BatchNormalization(momentum=0.95, trainable=train_mode)(x)
    # x = Activation('relu')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation='relu', name='pose_dense_6', trainable=train_mode)(x)

    # op 1
    x = BatchNormalization(momentum=0.94, trainable=train_mode)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='pose_dense_6', trainable=train_mode)(x)  # , kernel_regularizer='l2'
    x = Flatten(name='pose_flatten_2')(x)

    if USE_6POSE is True:
        out = Dense(6, activation='softmax', name='pose_dense_ouptut')(x)
    else:
        out = Dense(3, activation=None, name='pose_dense_ouptut', trainable=train_mode)(x)

    model = Model(inputs=base.inputs, outputs=out)

    if USE_ADAM_OPT is True:
        optimizer =  tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005) #  Adam(lr=0.05) tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
    else:
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    if train_mode:
        model.compile(optimizer, loss='mse', metrics=['accuracy', custom_acc])  # mse -> mean sqare error | 'accuracy'
    else:
        model.compile(optimizer, loss='mae', metrics=['accuracy', custom_acc])  # mse -> mean sqare error | 'accuracy' | mae -> mean absolute error

    model.summary()

    return model


def train(data: [Data], data_v: [Data], epochs=1, model_input=None, model_output=None):

    train_data_generator = DataGenerator(data, BATCH_SIZE, shuffle=True)
    valid_data_generator = DataGenerator(data_v, BATCH_SIZE)

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

    eval_data_generator = DataGenerator(data[:500], BATCH_SIZE)
    eval = model.evaluate_generator(eval_data_generator)

    log.info('train eval: %s' % eval)

    log.info('\nhistory dict:', history.history)

    bbox = data[0].bbox
    image_array = load_image(data[0].image, bbox)

    prediction = model.predict(np.array([image_array]))
    log.info(prediction)
    log.info(data[0].pose[:3])

    R_p, _ = cv2.Rodrigues(prediction[0])
    R_l, _ = cv2.Rodrigues(data[0].pose[:3])
    theta = np.arccos((np.trace(R_p.T @ R_l) - 1) / 2)
    log.info(np.rad2deg(theta))


def predict(data, model_input):
    model = get_model(train_mode=False)

    if model_input is not None:
        model.load_weights(model_input)

    # train_data_generator = DataGenerator(images[:1], [[0.110099974, 0.238761767, -0.443073971 ,40.78644871 ,34.50543871 ,1035.096866]], b_boxes[:1], 1)
    # model.fit_generator(train_data_generator)

    prediction_data_generator = PredictDataGenerator(data, BATCH_SIZE)
    data_generator = DataGenerator(data, BATCH_SIZE)
    predictions = model.predict_generator(prediction_data_generator, verbose=1)

    eval_loss = model.evaluate_generator(data_generator)
    log.info('predict eval loss: %s' % eval_loss)

    return predictions
