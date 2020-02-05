import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


def matplot_image(image: np.array):
    plt.figure()
    plt.imshow(image, interpolation='nearest')
    plt.show()


def pre_process_image(image: np.array, bbox: np.array):
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    img_h, img_w, img_c = image.shape
    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = bbox[2]
    rb_y = bbox[3]

    fillings = np.zeros((4, 1), dtype=np.int32)
    if lt_x < 0:  ## 0 for python
        fillings[0] = math.ceil(-lt_x)
    if lt_y < 0:
        fillings[1] = math.ceil(-lt_y)
    if rb_x > img_w - 1:
        fillings[2] = math.ceil(rb_x - img_w + 1)
    if rb_y > img_h - 1:
        fillings[3] = math.ceil(rb_y - img_h + 1)
    new_bbox = np.zeros((4, 1), dtype=np.float32)

    imgc = image
    if fillings[0] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack([np.zeros((img_h, fillings[0][0], img_c), dtype=np.uint8), imgc])
    if fillings[1] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack([np.zeros((fillings[1][0], img_w, img_c), dtype=np.uint8), imgc])
    if fillings[2] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack([imgc, np.zeros((img_h, fillings[2][0], img_c), dtype=np.uint8)])
    if fillings[3] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack([imgc, np.zeros((fillings[3][0], img_w, img_c), dtype=np.uint8)])

    new_bbox[0] = lt_x + fillings[0]
    new_bbox[1] = lt_y + fillings[1]
    new_bbox[2] = rb_x + fillings[0]
    new_bbox[3] = rb_y + fillings[1]

    # %% Crop
    bbox_new = np.ceil(new_bbox)
    side_length = max(bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1])
    bbox_new[2:4] = bbox_new[0:2] + side_length
    bbox_new = bbox_new.astype(int)

    crop_img = imgc[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0]]

    return crop_img


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def pre_process_image2(image, bbox, allow_upscale=True, max_width=224):
    x, y, w, h = bbox
    x, y, w, h = int(x - 1.), int(y - 1.), int(w + 1.), int(h + 1.)
    image = image[y: y + h, x: x + w]

    # in case bbox larger the image need to adjust
    h, w = image.shape[:2]

    # check if need to downscale
    if w > max_width or h > max_width:
        # rescale width
        if w > h:
            image = image_resize(image, width=max_width)
        else:
            image = image_resize(image, height=max_width)
    elif allow_upscale:
        # up scale
        if w > h:
            image = image_resize(image, width=max_width)
        else:
            image = image_resize(image, height=max_width)

    new_image = np.zeros((max_width, max_width, 3), dtype=np.uint8)
    new_image[:image.shape[0], :image.shape[1]] = image

    return new_image


def pre_process_image_naive(image, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x * 1.), int(y * 1.), int(w * 1.), int(h * 1.)
    image = image[y: y + h, x: x + w]

    return image


def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    # tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero,
                    tf.eye(3, batch_shape=[batch_size]), Rs)


if __name__ == '__main__':
    image_ = cv2.imread('../augmented/300w_3d_helen_naive_1/118737215_1_2_aug.jpg')
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    image_ = pre_process_image2(image_, [57, 228, 229, 230])
    matplot_image(image_)

