#
# Copyright (c) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Preprocessing methods.

This module provides methods for preprocessing images.

Todo:
    * Tests
        * ``convert_to_grayscale``
        * ``normalize``
        * ``pad_borders``
    * Fix and generalize ``distort_image``
"""
import tensorflow as tf


def convert_to_grayscale(image):
    """Convert RGB image to grayscale."""
    image = tf.image.rgb_to_grayscale(image)
    return image


def normalize(image):
    """Rescale pixels values (to [-1, 1])."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def pad_borders_or_shrink(image, char_bbox, line_bbox,
                          shape, maintain_aspect=True):
    """Pad or resize the image.

    If the desired shape is larger than the original, then that axis is
    padded equally on both sides with the mean pixel value in the image.
    Otherwise, the image is resized with BILINEAR interpolation such
    that the aspect ratio is maintained.

    Args:
        image (:obj:`tf.Tensor`): Image tensor [height, width, channels].
        char_bbox (:obj:`tf.Tensor`): Character bounding box [4].
        line_bbox (:obj:`tf.Tensor`): Line bounding box [4].
        shape (:obj:`tuple` of :obj:`int`): Output shape.
        maintain_aspect (bool): Maintain the aspect ratio.

    Returns:
        :obj:`tf.Tensor`: Resized image.
        :obj:`tf.Tensor`: Adjusted character bounding boxes.
        :obj:`tf.Tensor`: Adjusted line bounding boxes.
    """

    def shrink(h, w, ratio):
        def f1():
            scale = 1. / ratio
            new_h = h * scale
            new_w = w * scale
            return new_h, new_w

        def f2():
            return h, w

        h, w = tf.cond(tf.greater(ratio, 1), f1, f2)
        return h, w

    h_orig = tf.to_float(tf.shape(image)[0])
    w_orig = tf.to_float(tf.shape(image)[1])

    if maintain_aspect:
        # Shrink height
        h_ratio = tf.cast(h_orig / shape[0], tf.float32)
        height, width = shrink(h_orig, w_orig, h_ratio)

        # Shrink width
        w_ratio = tf.cast(width / shape[1], tf.float32)
        height, width = shrink(height, width, w_ratio)

        # Final resize
        image = tf.image.resize_images(
            image, size=[tf.to_int32(height), tf.to_int32(width)])
        h_ratio = tf.cast(height / shape[0], tf.float32)
        w_ratio = tf.cast(width / shape[1], tf.float32)
    else:
        height = h_orig
        width = w_orig
        h_ratio = tf.cast(tf.minimum(height / shape[0], 1), tf.float32)
        w_ratio = tf.cast(tf.minimum(width / shape[1], 1), tf.float32)
        image = tf.image.resize_images(image, size=shape)

    # Padding
    h_diff = tf.maximum(shape[0] - height, 0)
    w_diff = tf.maximum(shape[1] - width, 0)

    h_pad = h_diff / 2
    h_pad_ratio = tf.cast(h_pad / shape[0], tf.float32)
    h_pad = h_pad[tf.newaxis, tf.newaxis]
    w_pad = w_diff / 2
    w_pad_ratio = tf.cast(w_pad / shape[1], tf.float32)
    w_pad = w_pad[tf.newaxis, tf.newaxis]

    paddings = tf.concat(
        [tf.concat([tf.to_int32(h_pad), tf.to_int32(tf.ceil(h_pad))], axis=1),
         tf.concat([tf.to_int32(w_pad), tf.to_int32(tf.ceil(w_pad))], axis=1),
         tf.constant([[0, 0]])], axis=0
    )
    c = tf.reduce_mean(image)
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=c)

    # Correct Bounding boxes
    ratios = tf.stack([h_ratio, w_ratio, h_ratio, w_ratio])
    border = tf.stack([h_pad_ratio, w_pad_ratio, h_pad_ratio, w_pad_ratio])
    char_bbox_adjusted = tf.multiply(char_bbox, ratios) + border
    line_bbox_adjusted = tf.multiply(line_bbox, ratios) + border

    return image, char_bbox_adjusted, line_bbox_adjusted


def distort_image(image):
    # TODO: only works for single chars, broken if bounding boxes used
    # pad with average to avoid empty space padding
    image = tf.pad(tensor=image,
                   paddings=tf.constant([[16, 16], [16, 16], [0, 0]]),
                   constant_values=tf.reduce_mean(image))
    # rotate
    image = tf.contrib.image.rotate(
        image,
        tf.random_uniform(shape=[1],
                          minval=-0.3,
                          maxval=0.3)
    )
    # crop
    image = tf.image.central_crop(image, 2./3.)
    return image
