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

from carpedm.data.lang import JapaneseUnicodes
from carpedm.tasks.generic import Task
from carpedm.util import registry
from carpedm.util.eval import confusion_matrix_metric


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
        return JapaneseUnicodes('kana')

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

        accuracy = tf.metrics.accuracy(stacked_labels, predictions['classes'])
        metrics = {
            'accuracy': accuracy,
            'confusion': confusion_matrix_metric(
                stacked_labels, predictions['classes'], self.num_classes)
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


@registry.register_task
class OCRSeqKana3(OCRTask):

    def __init__(self, beam_width=100, **kwargs):
        self._beam_width = beam_width
        super(OCRSeqKana3, self).__init__(**kwargs)

    @property
    def character_set(self):
        return JapaneseUnicodes('kana')

    @property
    def image_scope(self):
        return 'seq'

    @property
    def sequence_length(self):
        return 3

    @property
    def sparse_labels(self):
        return True

    @property
    def target(self):
        return 'image/seq/char/id_sparse'

    def loss_fn(self, features, model_output, targets, is_training):
        return tf.nn.ctc_loss(labels=targets,
                              inputs=model_output['logits'],
                              sequence_length=model_output['seq_len'],
                              time_major=False)

    def results(self, loss, tower_features, tower_preds, tower_targets,
                is_training):

        tf.summary.image("sample_input", tower_features[0]['image/data'])

        all_logits = tf.concat([p['logits'] for p in tower_preds], axis=0)
        seq_lens = tf.concat([p['seq_len'] for p in tower_preds], axis=0)

        # TODO: fix when seqs are different lengths from multiple GPUs
        all_labels = tf.sparse_concat(0, [p for p in tower_targets])
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(all_logits, [1, 0, 2]),
            sequence_length=seq_lens,
            beam_width=self._beam_width)
        decoded = decoded[0]  # best path

        edit_distance = tf.edit_distance(decoded, tf.to_int64(all_labels),
                                         normalize=False)

        Z = tf.cast(tf.size(all_labels), tf.float32)
        ler = tf.reduce_sum(edit_distance) / Z
        S = tf.cast(tf.size(edit_distance), tf.float32)
        num_wrong_seqs = tf.cast(tf.count_nonzero(edit_distance), tf.float32)
        ser = num_wrong_seqs / S

        metrics = {
            'ler': tf.metrics.mean(ler),
            'ser': tf.metrics.mean(ser)
        }

        tensors_to_log = {'loss': loss, 'ler': ler, 'ser': ser}

        mapping_string = tf.constant(self._meta.vocab.types())
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_string, default_value='NULL')
        decoding = table.lookup(tf.to_int64(tf.sparse_tensor_to_dense(decoded)))
        gt = table.lookup(tf.to_int64(tf.sparse_tensor_to_dense(all_labels)))

        tf.summary.text('decoded', decoding)
        tf.summary.text('gt', gt)

        predictions = {
            'classes': tf.argmax(input=all_logits, axis=1),
            'probabilities': tf.nn.softmax(all_logits),
            'decoded': decoding,
        }

        return tensors_to_log, predictions, metrics

    def regularization(self, hparams):
        model_params = tf.trainable_variables()
        weight_loss = tf.multiply(
            hparams.weight_decay,
            tf.add_n([tf.nn.l2_loss(v) for v in model_params]),
            name='weight_loss')
        return weight_loss
