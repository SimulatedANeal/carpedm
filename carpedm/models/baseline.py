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
        x = features['image/data']
        x = self._cnn.forward_pass(
            x, data_format, axes_order, is_training, False, reuse)
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


@registry.register_model
class SequenceBaseline(TFModel):
    """A simple baseline CNN-LSTM model."""

    def __init__(self, num_classes, lstm_layers=2, lstm_units=100,
                 feature_extractor=nn.conv.CNN(), *args, **kwargs):
        """Initializer.

        Overrides TFModel.

        Args:
            num_classes (int): Number of possible character classes.
            lstm_layers (int): Number of LSTM layers.
            lstm_unit (int): Number of units in LSTM cell
            feature_extractor:
            *args: Unused arguments.
            **kwargs: Unused arguments.
        """
        self._num_classes = num_classes + 1  # Add CTC null label.
        self._layers = lstm_layers
        self._units = lstm_units
        self._feature_extractor = feature_extractor

    @property
    def name(self):
        return 'Baseline_seq_' + self._feature_extractor.name

    def _forward_pass(self, features, data_format, axes_order,
                      is_training, reuse):
        x = self._feature_extractor.forward_pass(
            features['image/data'], data_format, axes_order,
            is_training, False, reuse)
        if axes_order == [0, 3, 1, 2]:
            x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.reshape(x, [-1, x.shape[1], x.shape[2] * x.shape[3]])
        x = nn.rnn.bi_lstm(x, n_layers=self._layers, n_units=self._units)
        seq_len = tf.tile(tf.expand_dims(tf.to_int32(tf.shape(x)[1]), 0),
                          [tf.to_int32(tf.shape(x)[0])])
        logits = tf.layers.dense(inputs=x, units=self._num_classes)

        return {'logits': logits, 'seq_len': seq_len}

    def initialize_pretrained(self, pretrained_dir):

        submodel = 'Baseline_' + self._feature_extractor.name

        variable_mapping = dict()

        for i in range(5):
            variable_mapping[submodel + '/conv{}/'.format(i)] \
                = self.name + '/conv{}/'.format(i)

        return variable_mapping
