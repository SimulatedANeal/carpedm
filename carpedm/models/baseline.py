#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Baseline models."""

import tensorflow as tf

from carpedm.models.generic import TFModel
from carpedm import nn
from carpedm.util import registry


@registry.register_model
class SingleCharBaseline(TFModel):
    """A simple baseline CNN model."""

    def __init__(self, num_classes, *args, **kwargs):
        """Initializer.

        Overrides TFModel.

        Args:
            num_classes: Number of possible character classes.
            *args: Unused arguments.
            **kwargs: Unused arguments.

        """
        self._num_classes = num_classes
        self._cnn = nn.conv.CNN()

    @property
    def name(self):
        return "Baseline_" + self._cnn.name

    def _forward_pass(self, features, data_format, axes_order,
                      is_training, reuse):
        x = self._cnn.forward_pass(
            features, data_format, axes_order, is_training, False, reuse)
        x = tf.layers.flatten(x)
        tf.logging.info('image after flatten: %s', x.get_shape())

        x = tf.layers.dense(
            inputs=x, units=200, activation=tf.nn.relu, name='dense1')
        nn.util.activation_summary(x)
        x = tf.layers.dense(
            inputs=x, units=200, activation=tf.nn.relu, name='dense2')
        nn.util.activation_summary(x)
        logits = tf.layers.dense(
            inputs=x, units=self._num_classes, name='logits')
        return logits
