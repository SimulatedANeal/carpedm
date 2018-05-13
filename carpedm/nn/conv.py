#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Convolutional layers and components.

"""

import tensorflow as tf

from carpedm.models.generic import TFModel
from carpedm.util import registry
from carpedm.nn import util

@registry.register_model
class CNN(TFModel):
    """Modular convolutional neural network layer class."""

    def __init__(self,
                 kernel_size=((3, 3), (3, 3), (3, 3), (3, 3)),
                 num_filters=(64, 96, 128, 160),
                 padding='same',
                 pool_size=((2, 2), (2, 2), (2, 2), (2, 2)),
                 pool_stride=(2, 2, 2, 2),
                 pool_every_n=1,
                 pooling_fn=tf.layers.max_pooling2d,
                 activation_fn=tf.nn.relu,
                 *args,
                 **kwargs):
        """Initializer.

        Args:
            num_classes (int): Number of output classes.
            kernel_size: (tuple of (int, int)): Convolution kernel size
                for each layer.
            num_filters (tuple of int): Number of filters for
                corresponding convolution layer from kernel_size.
            padding (str): Type of padding to use at each conv layer.
            pool_size (tuple of (int, int)): Pooling size.
            pool_stride (tuple of int): Pooling stride.
            pool_every_n (int or None): Perform pooling every n nn.
            pooling_fn: Pooling function,
            activation_fn: Activation function.
            *args: Unused arguments
            **kwargs: Unused arguments
        """
        assert len(kernel_size) == len(num_filters), (
            "kernel_size and num_filters must be equal length"
        )
        assert len(pool_size) == len(pool_stride), (
            "pool_size and pool_stride must be equal length"
        )
        assert len(kernel_size) / pool_every_n == len(pool_size), (
            "{} / {} != {}".format(
                len(kernel_size), pool_every_n, len(pool_size))
        )
        self._kernels = kernel_size
        self._filters = num_filters
        self._pad = padding
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._pool_every_n = pool_every_n
        self._pool = pooling_fn
        self._activation = activation_fn
        super(CNN, self).__init__()

    @property
    def name(self):
        name = 'CNN_'
        p = 0
        for i in range(len(self._kernels)):
            name += "c{}.{}-".format(
                'x'.join(map(str, self._kernels[i])),
                self._filters[i])
            if self._pool_every_n and i % self._pool_every_n == 0:
                name += "p{}.{}-".format(
                    'x'.join(map(str, self._pool_size[p])),
                    self._pool_stride[p])
                p += 1
        return name[:-1]

    def _forward_pass(self, x, data_format, axes_order, is_training, reuse):
        x = tf.transpose(x, axes_order) if axes_order else x
        p = 0
        for c in range(len(self._kernels)):
            x = tf.layers.conv2d(
                inputs=x, filters=self._filters[c], name='conv%d' % c,
                kernel_size=self._kernels[c], activation=self._activation,
                padding=self._pad, data_format=data_format)
            util.activation_summary(x)
            tf.logging.info(
                'image after unit %s: %s', util.name_nice(x.op.name), x.get_shape())
            if self._pool_every_n and c % self._pool_every_n == 0:
                x = self._pool(
                    inputs=x, pool_size=self._pool_size[p], name='pool%d' % p,
                    strides=self._pool_stride[p], data_format=data_format)
                tf.logging.info(
                    'image after unit %s: %s',
                    util.name_nice(x.op.name), x.get_shape())
                p += 1
        return x
