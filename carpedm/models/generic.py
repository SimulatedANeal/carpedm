#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""This module defines base model classes."""

import abc

import tensorflow as tf


class Model(object):
    """Abstract class for models."""

    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def name(self):
        """Unique identifier for the model.

        Used to identify results generated with the model.

        Must be implemented by subclass.

        Returns:
            str: The model name.

        """

    @abc.abstractmethod
    def forward_pass(self, features, data_format, axes_order, is_training):
        """Main model functionality.

        Must be implemented by subclass.

        Args:
            features (array_like or dict): Input features.
            data_format (str): Image format expected for computation,
                'channels_last' (NHWC) or 'channels_first' (NCHW).
            axes_order (list or None): If not None, is a list defining
                the axes order to which image input should be transposed
                in order to match data_format.
            is_training (bool): Training if true, else evaluating.

        Returns:
            array_like or dict: The return value, e.g. class logits.

        """

    def initialize_pretrained(self, pretrained_dir):
        """Initialize a pre-trained model or sub-model.

        Args:
            pretrained_dir (str): Path to directory where pretrained
                model is stored. May be used to extract model/sub-model
                name. For example::

                    name = pretrained_dir.split('/')[-1].split('_')[0]

        Returns:
            dict: Map from pre-trained variable to model variable.

        """
        raise NotImplementedError


class TFModel(Model):
    """Abstract class for TensorFlow models."""

    @property
    @abc.abstractmethod
    def name(self):
        """Unique identifier for the model.

        The model name will serve as directory name for model-specific
        results and as the top-level ``tf.variable_scope``.

        Returns:
            str: The model name.

        """

    @abc.abstractmethod
    def _forward_pass(self, features, data_format, axes_order,
                      is_training, reuse):
        """Main model functionality.

        Must be implemented by subclass.
        """

    def forward_pass(self, features, data_format, axes_order, is_training,
                     new_var_scope=False, reuse=False):
        """Wrapper for making nested variable scopes.

        Extends Model.

        Args:
            new_var_scope (bool): Use a new variable scope.
            reuse (bool): Reuse variables with same scope.

        """
        if new_var_scope:
            with tf.variable_scope(self.name, reuse=reuse):
                return self._forward_pass(
                    features, data_format, axes_order, is_training, reuse)
        else:
            return self._forward_pass(
                features, data_format, axes_order, is_training, reuse)
