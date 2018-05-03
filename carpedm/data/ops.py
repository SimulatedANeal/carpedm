#
# Copyright (c) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.


"""Data operations.

This module contains several non-module-specific data operations.

Todo:
    * Tests
        * ``to_sequence_example``, ``parse_sequence_example``
        * ``sparsify_label``
        * ``shard_batch``
        * ``same_line``
        * ``ixs_in_region``
        * ``seq_norm_bbox_values``
"""
import tensorflow as tf


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting a bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature_list(values):
    """Wrapper for inserting int64 FeatureList into Example proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _float_feature_list(values):
    """Wrapper for inserting float Feature list into Example proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting bytes FeatureList into Example proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def to_sequence_example(feature_dict):
    """Convert features to TensorFlow SequenceExample.

    Args:
        feature_dict (dict): Dictionary of features.

    Returns:
        :obj:`tf.train.SequenceExample`
    """
    feature = dict()
    feature_list = dict()
    for key, value in feature_dict.items():
        if 'seq' in key:
            if 'bbox' in key:
                feature_list[key] = _float_feature_list(value)
            else:
                feature_list[key] = _int64_feature_list(value)
        elif 'data' in key:
            image_raw = value.tostring()
            feature[key] = _bytes_feature(image_raw)
        else:
            feature[key] = _int64_feature(value)
    context = tf.train.Features(feature=feature)
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(context=context,
                                       feature_lists=feature_lists)
    return example


def parse_sequence_example(serialized):
    """Parse a sequence example.

    Args:
        serialized (:obj:`tf.Tensor`): Serialized 0-D tensor of type
            string.

    Returns:
        dict: Dictionary of features.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            "image/data": tf.FixedLenFeature([], tf.string),
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/char/count": tf.FixedLenFeature([], tf.int64, 0),
            "image/line/count": tf.FixedLenFeature([], tf.int64, 0),
        },
        sequence_features={
            "image/seq/char/id": tf.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True),
            "image/seq/char/bbox/xmin": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/char/bbox/ymin": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/char/bbox/xmax": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/char/bbox/ymax": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/line/bbox/xmin": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/line/bbox/ymin": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/line/bbox/xmax": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
            "image/seq/line/bbox/ymax": tf.FixedLenSequenceFeature(
                [], tf.float32, allow_missing=True),
        }
    )

    feature_dict = {}
    height = tf.cast(context['image/height'], tf.int32)
    width = tf.cast(context['image/width'], tf.int32)
    for key, value in context.items():
        if 'data' in key:
            image = tf.decode_raw(value, tf.uint8)
            feature_dict[key] = tf.reshape(image, [height, width, 3])
        else:
            feature_dict[key] = tf.cast(value, tf.int32)
    for obj in ['char', 'line']:
        values = []
        for value in ['ymin', 'xmin', 'ymax', 'xmax']:
            key = "image/seq/{}/bbox/{}".format(obj, value)
            values.append(tf.reshape(sequence.pop(key), [-1, 1]))
        bboxes = tf.concat(axis=1, values=values)
        feature_dict['image/seq/' + obj + '/bbox'] = bboxes
    for key, value in sequence.items():
            feature_dict[key] = tf.cast(value, tf.int32)

    return feature_dict


def sparsify_label(label, length):
    """Convert a regular Tensor into a SparseTensor.

    Args:
        label (:obj:`tf.Tensor`): The label to convert.
        length (:obj:`tf.Tensor`): Length of the label

    Returns:
        tf.SparseTensor
    """
    length = tf.cast(length, dtype=tf.int64)
    indices = tf.where(tf.not_equal(label, 0))
    values = tf.cast(tf.gather_nd(label, indices), tf.int32)
    char_ids = tf.SparseTensor(indices, values, dense_shape=[length])
    return char_ids


def shard_batch(features, labels, batch_size, num_shards):
    """Shard a batch of examples.

    Args:
        features (dict): Dictionary of features.
        labels (:obj:`tf.Tensor`): labels
        batch_size (int): The batch size.
        num_shards (int): Number of shards into which batch is split.

    Returns:
        :obj:`list` of :obj:`dict`: Features as a list of dictionaries.
    """
    label_batch = tf.unstack(labels, num=batch_size, axis=0)
    label_shards = [[] for i in range(num_shards)]
    for i in range(batch_size):
        idx = i % num_shards
        label_shards[idx].append(label_batch[i])

    feature_shards = [{} for i in range(num_shards)]
    for key in features:
        feature_batch = tf.unstack(features[key], num=batch_size, axis=0)
        shards = [[] for i in range(num_shards)]
        for i in range(batch_size):
            idx = i % num_shards
            shards[idx].append(feature_batch[i])
        for i in range(num_shards):
            feature_shards[i][key] = tf.stack(shards[i])

    return feature_shards, label_shards


def in_line(xmin_line, xmax_line, ymin_line, xmin_new, xmax_new, ymax_new):
    """Heuristic for determining whether a character is in a line.

    Note:
        Currently dependent on the order in which characters are
        added. For example, a character may vertically overlap with a
        line, but adding it to the line would be out of reading order.
        This should be fixed in a future version.

    Args:
        xmin_line (:obj:`list` of :obj:`int`): Minimum x-coordinate of
            characters in the line the new character is tested against.
        xmax_line (:obj:`list` of :obj:`int`): Maximum x-coordinate of
            characters in the line the new character is tested against.
        ymin_line (int): Minimum y-coordinate of line the new character
            is tested against.
        xmin_new (int): Minimum x-coordinate of new character.
        xmax_new (int): Maximum x-coordinate of new character.
        ymax_new (int): Maximum y-coordinate of new character.

    Returns:
        bool:
            The new character vertically overlaps with the
            "average" character in the line.
    """
    xmin_avg = sum(xmin_line) / len(xmin_line)
    xmax_avg = sum(xmax_line) / len(xmax_line)
    return (xmin_avg <= xmax_new
            and xmax_avg >= xmin_new
            and ymax_new >= ymin_line)


def in_region(obj, region, entire=True):
    """Test if an object is in a region.

    Args:
        obj (tuple or BBox): Object bounding box
            (xmin, xmax, ymin, ymax) or point (x, y).
        region (tuple or BBox): Region (xmin, xmax, ymin, ymax).
        entire (bool): Object is entirely contained in region.

    Returns:
        bool: Result
    """
    if len(obj) == 4:
        if entire:
            result = (region[0] <= obj[0] <= obj[1] <= region[1]
                      and region[2] <= obj[2] <= obj[3] <= region[3])
        else:
            result = (region[0] <= obj[0] <= region[1]
                      or region[0] <= obj[1] <= region[1]
                      or region[2] <= obj[2] <= region[3]
                      or region[2] <= obj[3] <= region[3])
    else:
        assert len(obj) == 2, "Invalid point or bounding box."
        result = (region[0] <= obj[0] <= region[1]
                  and region[2] <= obj[1] <= region[3])
    return result


def ixs_in_region(bboxes, y1, y2, x1, x2):
    """Heuristic for determining objects in a region.

    Args:
        bboxes (:obj:`list` of :obj:`carpedm.data.util.BBox`): Bounding
            boxes for object boundaries.
        y1 (int): Top (lowest row index) of region.
        y2 (int): Bottom (highest row index) of region.
        x1 (int): left side (lowest column index) of region.
        x2 (int): right side (highest column index) of region.

    Returns:
        :obj:`list` of :obj:`int`: Indices of objects inside region.
    """
    result = []
    for i in range(len(bboxes)):
        b = bboxes[i]
        if (b.xmin >= x1 and b.xmax <= x2
                and y1 <= b.ymin and b.ymax <= y2):
            result.append(i)
    return result


def seq_norm_bbox_values(bboxes, height, width):
    """Sequence and normalize bounding box values.

    Args:
        bboxes (:obj:`list` of :obj:`carpedm.data.util.BBox`):
            Bounding boxes to process.
        width (int): Width (in pixels) of image bboxes are in.
        height (int): Height (in pixels) of image bboxes are in.

    Returns:
        tuple: :obj:`tuple` containing:

            :obj:`list` of :obj:`float`: Normalized minimum x-values

            :obj:`list` of :obj:`float`: Normalized minimum y-values

            :obj:`list` of :obj:`float`: Normalized maximum x-values

            :obj:`list` of :obj:`float`: Normalized maximum y-values
    """
    xmin, ymin, xmax, ymax = [], [], [], []
    for b in bboxes:
        xmin.append(b.xmin / width)
        ymin.append(b.ymin / height)
        xmax.append(b.xmax / width)
        ymax.append(b.ymax / height)
    return xmin, ymin, xmax, ymax
