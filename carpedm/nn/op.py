#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Operations for transforming network layer or input."""

import tensorflow as tf


# def apply_sliding_window(image, size, stride, data_format):
#     """Apply a sliding window to image to extract patches.
#
#     Args:
#         image:
#         size:
#         stride:
#         data_format:
#
#     Returns:
#
#     """
#     # tf.extract_image_patches expects batches
#     if len(image.get_shape().as_list()) == 3:
#         image = tf.expand_dims(image, axis=0)
#
#     if data_format == 'channels_first':
#         image = tf.transpose(image, [0, 2, 3, 1])
#
#     windows = tf.extract_image_patches(
#         image,
#         ksizes=[1, size[0], size[1], 1],
#         strides=[1, stride, image.shape[2], 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
