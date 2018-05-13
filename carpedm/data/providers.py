#
# Copyright (c) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.


"""Data providers for Task input function.

This module provides a generic interface for providing data useable
by machine learning algorithms.

A provider may either (1) receive data from the method that initialized
it, or (2) receive a directory path where the data to load is stored.

Todo:
    * Generator
        * numpy
        * pandas DataFrame

"""
import os
import functools
import abc

import tensorflow as tf

from carpedm.data import ops, preproc


class DataProvider(object):
    """Data provider abstract class."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, target_id):
        """Initializer.

        Args:
            target_id (str): Feature to use as the target value.

        """
        self._target_id = target_id

    @abc.abstractmethod
    def make_batch(self, batch_size):
        """Generator method that returns a new batch with each call.

        Args:
            batch_size (int): Number of examples per batch.

        Returns:
            dict: Batch features.
            array_like: Batch targets.

        """

    @property
    @abc.abstractmethod
    def format(self):
        return 'channels_last'


class TFDataSet(DataProvider):
    """TensorFlow DataSet provider from TFRecords stored on disk."""

    def __init__(self,
                 target_id,
                 data_dir,
                 subset,
                 num_examples,
                 pad_shape,
                 sparse_labels):
        """Initializer.

        Extends DataProvider.

        Args:
            data_dir (str): Directory containing (sharded) tfrecord
                files.
            subset (str): One of {'train', 'dev', 'test'}.
            num_examples (int): Number of examples in subset.
            pad_shape (tuple): Shape (height, width) of padded images.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.num_examples = num_examples
        self.pad_shape = pad_shape
        self.sparse_labels = sparse_labels
        super(TFDataSet, self).__init__(target_id)
        channels = tf.Dimension(1)  # Converting to grayscale in _preproc
        self._padding = {
            "image/data": tf.TensorShape([tf.Dimension(self.pad_shape[0]),
                                          tf.Dimension(self.pad_shape[1]),
                                          channels]),
            "image/height": tf.TensorShape([]),
            "image/width": tf.TensorShape([]),
            "image/char/count": tf.TensorShape([]),
            "image/line/count": tf.TensorShape([]),
            "image/mask/char": tf.TensorShape([
                tf.Dimension(self.pad_shape[0]),
                tf.Dimension(self.pad_shape[1]), 1
            ]),
            "image/mask/line": tf.TensorShape([
                tf.Dimension(self.pad_shape[0]),
                tf.Dimension(self.pad_shape[1]), 1
            ]),
            "image/seq/char/id": tf.TensorShape([None]),
            "image/seq/char/id_sparse": tf.TensorShape([None]),
            "image/seq/char/bbox": tf.TensorShape([None, 4]),
            "image/seq/line/bbox": tf.TensorShape([None, 4]),
        }

    @property
    def format(self):
        return 'channels_last'

    def make_batch(self, batch_size, single_char=False):
        filenames = self._get_filenames()
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        dataset = dataset.map(
            functools.partial(self._parser,
                              distort=(single_char and
                                       self.subset == 'train')),
            num_parallel_calls=batch_size)

        if self.subset == 'train':
            min_q_exs = 0.4 * self.num_examples
            dataset = dataset.shuffle(
                buffer_size=int(min_q_exs + 3 * batch_size)
            )
        padded_shapes = tuple([self._padding[k] for k in self.feat_keys])
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        features = dict(zip(self.feat_keys, batch))
        for key, value in features.items():
            if 'sparse' in key:
                features[key] = tf.deserialize_many_sparse(value,
                                                           dtype=tf.int32)
        labels = features.pop(self._target_id, None)

        return features, labels

    def _get_filenames(self):
        if self.subset in ['train', 'dev', 'test']:
            files = os.listdir(self.data_dir)
            relevant = [os.path.join(self.data_dir, f)
                        for f in files if self.subset in f]
            return relevant
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def _parser(self, serialized, distort=False):
        tensor_dict = ops.parse_sequence_example(serialized)
        tensor_dict['image/data'] = self._preproc(tensor_dict['image/data'])
        (tensor_dict['image/data'],
         tensor_dict['image/seq/char/bbox'],
         tensor_dict['image/seq/line/bbox']) = preproc.pad_borders_or_shrink(
            tensor_dict['image/data'], tensor_dict['image/seq/char/bbox'],
            tensor_dict['image/seq/line/bbox'], self.pad_shape)

        if self.sparse_labels:
            tensor_dict['image/seq/char/id_sparse'] = tf.serialize_sparse(
                ops.sparsify_label(tensor_dict['image/seq/char/id'],
                                   tensor_dict['image/char/count'])
            )

        # if distort: image = distort_image(image)
        self.feat_keys, features = zip(*tensor_dict.items())
        return features

    def _preproc(self, image):
        image = preproc.normalize(preproc.convert_to_grayscale(image))
        return image
