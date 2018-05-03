#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.


"""Optical character recognition tasks.

TODO:
    * Modularize common loss functions, select by id
    * Modularize common regularization options, select by id
"""
import abc

import tensorflow as tf

from carpedm.tasks.generic import Task
from carpedm.util import registry
from carpedm.util.eval import confusion_matrix


class OCRTask(Task):
    """Abstract class for OCR Tasks."""

    def __init__(self, **kwargs):
        super(OCRTask, self).__init__(**kwargs)

    @property
    def target(self):
        return 'image/seq/char/id'

    @property
    def blocks(self):
        return False

    @property
    def character(self):
        return True

    @property
    def line(self):
        return False

    @property
    def label(self):
        return True

    @property
    def bbox(self):
        return False

    @property
    @abc.abstractmethod
    def sparse_labels(self):
        return False

    def regularization(self, hparams):
        raise NotImplementedError

    def results(self, loss, tower_features, tower_preds, tower_targets,
                is_training):
        raise NotImplementedError

    def loss_fn(self, features, model_output, targets, is_training):
        raise NotImplementedError


@registry.register_task
class OCRSingleKana(OCRTask):
    """Single character recognition tasks."""

    @property
    def image_scope(self):
        return 'char'

    @property
    def character_set(self):
        return 'kana'

    def results(self, loss, tower_features, tower_preds, tower_targets,
                is_training):
        tensors_to_log = {'loss': loss}

        tf.summary.image("sample_input", tower_features[0]['image/data'])

        all_logits = tf.concat([p for p in tower_preds], axis=0)
        predictions = {
            'classes': tf.argmax(all_logits, axis=1),
            'probabilities': tf.nn.softmax(all_logits)
        }

        stacked_labels = tf.squeeze(tf.concat(tower_targets, axis=0))
        confusion_matrix(
            stacked_labels, predictions['classes'], self.num_classes)
        accuracy = tf.metrics.accuracy(stacked_labels, predictions['classes'])
        metrics = {
            'accuracy' if is_training else 'accuracy': accuracy
        }

        return tensors_to_log, predictions, metrics

    def loss_fn(self, features, model_output, targets, is_training):
        with tf.name_scope('batch_xentropy'):
            loss = tf.losses.sparse_softmax_cross_entropy(
                logits=model_output, labels=targets)
        return loss

    def regularization(self, hparams):
        model_params = tf.trainable_variables()
        weight_loss = tf.multiply(
            hparams.weight_decay,
            tf.add_n([tf.nn.l2_loss(v) for v in model_params]),
            name='weight_loss')
        return weight_loss

    @property
    def sparse_labels(self):
        return False
