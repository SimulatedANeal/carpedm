#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Evaluation helpers."""
import tensorflow as tf


def confusion_matrix(labels, predictions, num_classes):
    """Generate confusion matrix summary.

    Args:
        labels: Ground truth labels.
        predictions: Predictions.
        num_classes: Total number of classes.

    Returns:
        nothing
    """
    cm = tf.confusion_matrix(
        tf.reshape(labels, [-1]),
        tf.reshape(predictions, [-1]),
        num_classes
    )
    cm = cm / tf.reduce_max(cm)
    tf.summary.image("confusion", tf.expand_dims(tf.expand_dims(cm, 2), 0))
