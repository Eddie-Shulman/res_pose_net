import datetime
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
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling3D, AveragePooling2D, \
    Flatten, Dropout, Activation, GlobalMaxPooling2D, Conv2D, MaxPooling2D, ZeroPadding2D
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


def load_image(data: Data):
    image_array = cv2.imread(data.image)  # BGR
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    if data.bbox is not None:
        try:
            image_array = Utils.pre_process_image2(image_array, data.bbox)
        except:
            log.exception('failed to pre process image %s with bbox %s' % (data.image, data.bbox))
            raise Exception('failed to pre process image %s with bbox %s' % (data.image, data.bbox))
    else:
        raise Exception('no bbox data for image %s' % data.image)
    image_array = cv2.resize(image_array, (RESNET_SIZE, RESNET_SIZE), interpolation=cv2.INTER_CUBIC)
    image_array = np.asarray(image_array, dtype=np.float32)
    image_array = preprocess_input(image_array, mode='caffe')
    return image_array


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data: [Utils.Data], batch_size, shuffle=False) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        # initial data shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        idx = idx * self.batch_size
        images, poses = [], []

        for i in range(idx, idx + self.batch_size):
            data = self.data[i]
            image_array = load_image(data)
            images.append(image_array)
            pose = data.pose if USE_6POSE is True else data.pose[:3]
            poses.append(np.array(pose, dtype=np.float32))

        return np.array(images), np.array(poses, dtype=np.float32)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.shuffle:
            np.random.shuffle(self.data)


class PredictDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, idx):
        idx = idx * self.batch_size
        images = []
        for i in range(idx, idx + self.batch_size):
            data = self.data[i]
            image_array = load_image(data)
            images.append(image_array)

        return np.array(images)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))


def custom_acc(y_true, y_pred):
    mean_theta = tf.reduce_mean(tf.math.reduce_euclidean_norm(y_true - y_pred, 1))
    # mean_theta = custom_loss_3(y_true, y_pred)
    # mean_theta = custom_loss(y_true, y_pred)
    correct_prediction = (mean_theta * 180. / np.pi)
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


def get_custom_model(input, train_mode=True):
    x = input
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization(momentum=0.94, trainable=train_mode)(x)
    x = Activation('relu')(x)
    x = Dense(1024, activation='relu', name='pose_dense_1', trainable=train_mode)(x)  # , kernel_regularizer='l2'
    x = Flatten(name='pose_flatten_2')(x)

    return x


def get_custom_model2(input, train_mode=True):
    x = input
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad', trainable=train_mode)(x)
    x = Conv2D(18, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', strides=(2, 2), trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', strides=(2, 2), trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', strides=(2, 2), trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', strides=(2, 2), trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', strides=(2, 2), trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', trainable=train_mode)(x)
    x = BatchNormalization(trainable=train_mode, axis=-1)(x)
    x = Activation('relu')(x)

    if train_mode:
        x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu', name='pose_dense_1', trainable=train_mode)(x)  # , kernel_regularizer='l2'
    x = Flatten(name='pose_flatten_2')(x)

    # x = BatchNormalization(momentum=0.94, trainable=train_mode)(x)
    # x = Activation('relu')(x)
    if train_mode:
        x = Dropout(0.5)(x)

    return x


def get_resnet_transfer_model(input, train_mode=True, freeze_reznet=True):
    base = ResNet50(input_tensor=input, include_top=False, weights='imagenet')

    if freeze_reznet:
        for layer in base.layers:
            # freeze all but the last (5th) and half (inner) of the 4th layers
            if not (layer.name.startswith('bn5') or layer.name.startswith('res5') or
                    layer.name.startswith('bn4d') or layer.name.startswith('res4d') or
                    layer.name.startswith('bn4e') or layer.name.startswith('res4e') or
                    layer.name.startswith('bn4f') or layer.name.startswith('res4f')):
                layer.trainable = False
            else:
                log.info('keep layer %s trainable' % layer.name)

    x = base.output
    x = BatchNormalization(momentum=0.94, trainable=train_mode)(x)
    x = Activation('relu')(x)
    if train_mode:
        x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu', name='pose_dense_6', trainable=train_mode)(x)  # , kernel_regularizer='l2'
    x = Flatten(name='pose_flatten_2')(x)
    if train_mode:
        x = Dropout(0.5)(x)

    return x


def get_model(train_mode=True):

    input = Input(shape=(RESNET_SIZE, RESNET_SIZE, 3))
    # x = get_custom_model(input, train_mode=train_mode)
    x = get_custom_model2(input, train_mode=train_mode)
    # x = get_resnet_transfer_model(input, train_mode=train_mode, freeze_reznet=True)

    if USE_6POSE is True:
        out = Dense(6, activation='softmax', name='pose_dense_ouptut', trainable=train_mode)(x)
    else:
        out = Dense(3, activation=None, name='pose_dense_ouptut', trainable=train_mode)(x)

    model = Model(inputs=input, outputs=out)

    if USE_ADAM_OPT is True:
        optimizer =  tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005) #  Adam(lr=0.05) tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
    else:
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.001, momentum=0.3)

    if train_mode:
        model.compile(optimizer, loss='mse', metrics=['accuracy', 'mae', custom_acc])  # mse -> mean sqare error | 'accuracy'
    else:
        model.compile(optimizer, loss='mae', metrics=['accuracy', 'mae', custom_acc])  # mse -> mean sqare error | 'accuracy' | mae -> mean absolute error

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

    log_dir = os.path.join('..', 'trian_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    callbacks.append(tensorboard_callback)

    history = model.fit_generator(train_data_generator,
                                  epochs=epochs, verbose=1,
                                  validation_data=valid_data_generator,
                                  callbacks=callbacks
                                  )

    eval_data_generator = DataGenerator(data[:500], BATCH_SIZE)
    eval = model.evaluate_generator(eval_data_generator)
    log.info('train eval: %s' % eval)

    eval_data_generator = DataGenerator(data_v, BATCH_SIZE)
    eval = model.evaluate_generator(eval_data_generator)
    log.info('validation eval: %s' % eval)

    log.info('\nhistory dict:', history.history)

    image_array = load_image(data[0])

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


def validate_load_image():
    import DataSources
    from detect_face import DetectFace

    from keras.preprocessing import image
    # from keras.applications.resnet50 import preprocess_input

    data = DataSources.load_validation_dataset2()
    data: [Data] = DetectFace.get_face_bboxes(data[:1])

    image_array = load_image(data[0])
    image_array = preprocess_input(image_array, mode='tf')


    img = image.load_img(data[0].image, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    print('done')


def validate_pose_vs_landmarks():
    import DataSources
    import GenerateTrainingSet

    data = DataSources.load_naive_augmented_dataset(DataSources.DataSources.VALIDATION_SET2_NG)
    data = DataSources.load_validation_dataset2(DataSources.DataSources.VALIDATION_2, recalc_pose=False)

    total_theta = 0
    for data_ in data:
        face_model = GenerateTrainingSet.get_face_model()
        rot_mat_orig, _ = cv2.Rodrigues(data_.pose[:3])
        rotation_vecs, translation_vecs = GenerateTrainingSet.solve_pnp(data_.landmarks_2d, face_model.model_TD, face_model)
        rot_mat_land, _ = cv2.Rodrigues(rotation_vecs)

        theta = Utils.get_theta_between_rot_mats(rot_mat_orig, rot_mat_land)
        total_theta += theta

    print(np.rad2deg(total_theta / len(data)))


if __name__=='__main__':
    validate_pose_vs_landmarks()


