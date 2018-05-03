#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
#
# Portions of this module are based on or taken from the TensorFlow
# models "im2text" data pipeline, so here is their license.
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Data utilities.

This module provides utility methods/classes used by other data modules.

Todo:
    * Tests
        * ``generate_features``
    * Refactor ``generate_features``
    * Fix ``class_mask`` for overlapping characters.
"""
import itertools
import os
from re import match

import numpy as np
import tensorflow as tf

from carpedm.data import ops


def image_path(data_dir, bib_id, image_id):
    """Generate path to a specified image.

    Args:
        data_dir (str): Path to top-level data directory.
        bib_id (str): Bibliography ID.
        image_id (str): Image ID.

    Returns: String

    """
    return os.path.join(data_dir, bib_id, 'images', image_id + '.jpg')


class BBox(object):
    """Bounding box helper class."""

    def __init__(self, xmin, xmax, ymin, ymax):
        """Initializer.

        Args:
            xmin:
            xmax:
            ymin:
            ymax:
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.box = (self.xmin, self.xmax, self.ymin, self.ymax)

    def __getitem__(self, item):
        return self.box[item]

    def __len__(self):
        return len(self.box)


class Character(object):
    """Helper class for storing a single character."""

    def __init__(self, label, image_id, x, y, block_id, char_id, w, h):
        """Initializer.

        Argument order matches csv format.

        Args:
            label (str): Unicode-like label for the character.
            image_id (str): Identifier (e.g. filepath) for
                image from which the character comes.
            x (str or int): X-coordinate (column) of character's top-
                left corner, relative to left (col[0]) of parent image.
            y (str or int): Y-coordinate (row) of character's top-left
                corner, relative to top (row[0]) of parent image.
            block_id (str): ID for the character's block.
            char_id (str): Unique ID for character (token) in an image.
            w (str or int): Width (in pixels).
            h (str or int): Height (in pixels).

        """
        assert match(r'^U\+[0-9A-Fa-f]{4,5}$', label), (
            "Invalid label %s" % label
        )
        assert match(r'^B[0-9]{4}$', block_id), (
            "Invalid block ID %s" % block_id
        )
        assert match(r'^C[0-9]{4}$', char_id), (
            "Invalid character ID %s" % char_id
        )
        self.label = label
        self.image_id = image_id
        self.block_id = block_id
        self.id = char_id
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)


class ImageTFOps(object):
    """Helper class for decoding and resizing images."""

    _sess = tf.Session()
    _encoded_jpeg = tf.placeholder(dtype=tf.string)
    _decode_jpeg = tf.image.decode_jpeg(_encoded_jpeg, channels=3)

    _image_orig = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3))
    _shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    _image_resize = tf.cast(tf.image.resize_images(_image_orig, size=_shape),
                            tf.uint8)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def resize(self, image, shape):
        image = self._sess.run(
            self._image_resize,
            feed_dict={
                self._image_orig: image,
                self._shape: shape
            }
        )
        return image


class ImageMeta(object):
    """Class for storing and manipulating image metadata."""

    _image_helper = ImageTFOps()

    def __init__(self, filepath, full_image=False, first_char=None):
        """Initializer

        Args:
            filepath (str): Path to parent image.
            full_image (bool): Use full parent image.
            first_char (Character or None): First character.

        """
        self.filepath = filepath
        self._full = full_image
        self._labels = []
        self._blocks = []
        self._ids = []
        self._x_raw = []
        self._y_raw = []
        self._w_raw = []
        self._h_raw = []
        self._w_full = None
        self._h_full = None
        self._out_shape = (None, None)
        if first_char:
            self._x_raw.append(first_char.x)
            self._y_raw.append(first_char.y)
            self._w_raw.append(first_char.w)
            self._h_raw.append(first_char.h)
            self._labels.append(first_char.label)
            self._blocks.append(first_char.block_id)
            self._ids.append(first_char.id)

    @property
    def full_w(self):
        """Width (in pixels) of full raw parent image.

        Returns:
            int: The return value.

        """
        if self._w_full is None:
            im = self._load_image()
            self._w_full = im.shape[1]
            self._h_full = im.shape[0]
        return self._w_full

    @property
    def full_h(self):
        """Height (in pixels) of full raw parent image.

        Returns:
            int: The return value.

        """
        if self._h_full is None:
            im = self._load_image()
            self._h_full = im.shape[0]
            self._w_full = im.shape[1]
        return self._h_full

    @property
    def xmin(self):
        """Image's minimum x-coordinate (column) in raw parent image.

        Returns:
            int: The return value.

        """
        if self._full or len(self._x_raw) == 0:
            return 0
        else:
            return min(self._x_raw)

    @property
    def xmax(self):
        """Image's maximum x-coordinate (column) in raw parent image.

        Returns:
            int: The return value.

        """
        if self._full or len(self._char_xmax) == 0:
            return self.full_w
        else:
            return max(self._char_xmax)

    @property
    def ymin(self):
        """Image's minimum y-coordinate (row) in raw parent image.

        Returns:
            int: The return value.

        """
        if self._full or len(self._y_raw) == 0:
            return 0
        else:
            return min(self._y_raw)

    @property
    def ymax(self):
        """Image's maximum y-coordinate (row) in raw parent image.

        Returns:
            int: The return value.

        """
        if self._full or len(self._char_ymax) == 0:
            return self.full_h
        else:
            return max(self._char_ymax)

    @property
    def width(self):
        """Width (in pixels) in full parent image original scale.

        Returns:
            int: The return value.

        """
        return self.xmax - self.xmin

    @property
    def height(self):
        """Height (in pixels) in full parent image original scale.

        Returns:
            int: The return value.

        """
        return self.ymax - self.ymin

    @property
    def num_chars(self):
        """Number of characters in the image.

        Returns:
            int: The return value.

        """
        return len(self._labels)

    @property
    def char_labels(self):
        """Character labels

        Returns:
            :obj:`list` of :obj:`str`: The return value.

        """
        return self._labels

    @property
    def char_bboxes(self):
        """Bounding boxes for characters.

        Returned bounding boxes are relative to
        (:meth:`xmin`, :meth:`ymin`).

        Returns:
            :obj:`list` of :obj:`carpedm.data.util.BBox`:
                The return values.

        """
        scale_h, scale_w = self.new_shape(self._out_shape, ratio=True)
        adjusted_bboxes = [
            BBox(xmin=(self._x_raw[i] - self.xmin) * scale_w,
                 xmax=(self._char_xmax[i] - self.xmin) * scale_w,
                 ymin=(self._y_raw[i] - self.ymin) * scale_h,
                 ymax=(self._char_ymax[i] - self.ymin) * scale_h)
            for i in range(self.num_chars)
        ]
        return adjusted_bboxes

    @property
    def line_bboxes(self):
        """Bounding boxes for lines in the image,

        Note: Currently only meaningful when using full page image.

        Returns:
            :obj:`list` of :obj:`BBox`: The return values.

        """
        result = []
        if self._full:
            bboxes = self.char_bboxes
            b = bboxes[0]
            xmin, xmax, ymin, ymax = [b.xmin], [b.xmax], [b.ymin], [b.ymax]
            for b in bboxes[1:]:
                if not ops.in_line(xmin_line=xmin,
                                   xmax_line=xmax,
                                   ymin_line=min(ymin),
                                   xmin_new=b.xmin,
                                   xmax_new=b.xmax,
                                   ymax_new=b.ymax):
                    result.append(BBox(min(xmin), max(xmax),
                                       min(ymin), max(ymax)))
                    xmin, xmax = [b.xmin], [b.xmax]
                    ymin, ymax = [b.ymin], [b.ymax]
                else:
                    xmin.append(b.xmin)
                    xmax.append(b.xmax)
                    ymin.append(b.ymin)
                    ymax.append(b.ymax)
            # Add last line
            result.append(BBox(min(xmin), max(xmax), min(ymin), max(ymax)))
        return result

    @property
    def char_mask(self):
        """Generate pseudo-pixel-level character mask.

        Pixels within character bounding boxes are assigned to positive
        class (1), others assigned negative class (0).

        Returns:
            :obj:`numpy.ndarray`: Character mask of shape (height, width, 1)

        """
        mask = np.zeros(self._out_shape[:2])
        for b in self.char_bboxes:
            mask[b.ymin:b.ymax, b.xmin:b.xmax] = 1
        mask = np.expand_dims(mask, 2)
        return mask.astype(dtype=np.float32)

    @property
    def line_mask(self):
        """Generate pseudo-pixel-level line mask.

        Pixels within line bounding boxes are assigned to positive
        class (1), others assigned negative class (0).

        Returns:
            :obj:`numpy.ndarray`: Line mask of shape (height, width, 1)

        """
        mask = np.zeros(self._out_shape[:2])
        for b in self.line_bboxes:
            mask[b.ymin:b.ymax, b.xmin:b.xmax] = 1
        mask = np.expand_dims(mask, 2)
        return mask.astype(dtype=np.float32)

    def class_mask(self, vocab):
        """Generate a character class image mask.

        Note:
            Where characters overlap, the last character added is
            arbitrarily the one that will be represented in the mask.
            This should be fixed in a future version.

        Args:
            vocab (Vocabulary): The vocabulary for converting to ID.

        Returns:
            :obj:`numpy.ndarray`: Class mask of shape (height, width, 1)

        """
        mask = np.zeros(self._out_shape[:2])
        for label, b in zip(self.char_labels, self.char_bboxes):
            mask[b.ymin:b.ymax, b.xmin:b.xmax] = vocab.char_to_id(label)

        mask = np.expand_dims(mask, 2)
        return mask.astype(dtype=np.float32)

    def generate_features(self,
                          image_shape,
                          vocab,
                          chunk,
                          character,
                          line,
                          label,
                          bbox):
        """

        Args:
            image_shape (tuple or None): Shape (height, width) to which
                images are resized, or the size of each chunk if
                chunks == True.
            vocab (Vocabulary or None): Vocabulary for converting
                characters to IDs. Required ``if character and label``.
            chunk (bool): Instead of using the original image, return
                a list of image chunks and corresponding features
                extracted from the original image on a regular grid.
                The original image is padded to divide evenly by chunk
                shape.
            character (bool): Include character info (ID, bbox).
            line (bool): Include line info (bbox) in features.
            label (bool): Include label IDs in features.
            bbox (str or None): If not None, include bbox in features
                as unit (e.g. 'pixel', 'ratio' [of image]))

        Returns:
            :obj:`list` of :obj:`dict`: Feature dictionaries.

        """
        if character and label:
            assert vocab, "Must provide vocab."

        features = {}
        shape = self.new_shape(image_shape)
        full_shape = None if chunk else shape
        im = self.load_image(full_shape)
        h, w, c = im.shape
        features['image/data'] = im
        features['image/height'] = h
        features['image/width'] = w
        features['image/channels'] = c
        if character:
            features['image/char/count'] = self.num_chars
            if label:
                features['image/seq/char/id'] = [vocab.char_to_id(c)
                                                 for c in self.char_labels]
            if bbox:
                bboxes = self.char_bboxes
                xmin, ymin, xmax, ymax = ops.seq_norm_bbox_values(bboxes, h, w)
                features['image/seq/char/bbox/xmin'] = xmin
                features['image/seq/char/bbox/ymin'] = ymin
                features['image/seq/char/bbox/xmax'] = xmax
                features['image/seq/char/bbox/ymax'] = ymax
        if line and self._full:
            if bbox:
                bboxes = self.line_bboxes
                features['image/line/count'] = len(bboxes)
                xmin, ymin, xmax, ymax = ops.seq_norm_bbox_values(bboxes, h, w)
                features['image/seq/line/bbox/xmin'] = xmin
                features['image/seq/line/bbox/ymin'] = ymin
                features['image/seq/line/bbox/xmax'] = xmax
                features['image/seq/line/bbox/ymax'] = ymax
        if chunk:
            result = []
            img = features['image/data']
            h_diff = features['image/height'] % shape[0]
            w_diff = features['image/width'] % shape[1]
            padding = ((0, shape[0] - h_diff), (0, shape[1] - w_diff), (0, 0))
            img = np.pad(img, pad_width=padding, mode='mean')
            ys = np.arange(0, img.shape[0], shape[0])
            xs = np.arange(0, img.shape[1], shape[1])
            # top left of each block
            coordinates = list(itertools.product(ys, xs))
            for coord in coordinates:
                y1, y2 = coord[0], coord[0] + shape[0]
                x1, x2 = coord[1], coord[1] + shape[1]
                region = dict()
                region['image/data'] = img[y1:y2, x1:x2, :]
                region['image/height'] = shape[0]
                region['image/width'] = shape[1]
                region['image/channels'] = img.shape[2]
                if character:
                    char_ixs = ops.ixs_in_region(
                        features['image/seq/char/bbox'], y1, y2, x1, x2)
                    characters = list(map(lambda i: self.char_labels[i],
                                          char_ixs))
                    if label:
                        region['image/seq/char/id'] = [vocab.char_to_id(c)
                                                       for c in characters]
                        region['image/char/count'] = len(char_ixs)
                    if bbox:
                        bboxes = list(map(lambda i:
                                          features['image/seq/char/bbox'][i],
                                          char_ixs))
                        xmin, ymin, xmax, ymax = ops.seq_norm_bbox_values(
                            bboxes, height=shape[0], width=shape[1])
                        region['image/seq/char/bbox/xmin'] = xmin
                        region['image/seq/char/bbox/ymin'] = ymin
                        region['image/seq/char/bbox/xmax'] = xmax
                        region['image/seq/char/bbox/ymax'] = ymax

                if line and self._full:
                    line_ixs = ops.ixs_in_region(features['image/seq/line/bbox'],
                                                 y1, y2, x1, x2)
                    if bbox:
                        bboxes = list(map(
                            lambda i: features['image/seq/line/bbox'][i],
                            line_ixs))
                        xmin, ymin, xmax, ymax = ops.seq_norm_bbox_values(
                            bboxes, height=shape[0], width=shape[1])
                        region['image/seq/char/bbox/xmin'] = xmin
                        region['image/seq/char/bbox/ymin'] = ymin
                        region['image/seq/char/bbox/xmax'] = xmax
                        region['image/seq/char/bbox/ymax'] = ymax
                result.append(region)
        else:
            result = [features]
        return result

    def load_image(self, shape):
        """Load image and resize to shape.

        If ``shape`` is None or (None, None), original size is
        maintained.

        Args:
            shape (tuple or None): Output dimensions (height, width).

        Returns:
            :obj:`numpy.ndarray`: Resized image.

        """
        if shape:
            assert len(shape) == 2
        new_shape = self.new_shape(shape)

        image = self._load_image()
        image = image[self.ymin:self.ymax, self.xmin:self.xmax, :]
        image = self._image_helper.resize(image, new_shape)

        image = np.array(image, dtype=np.uint8)
        self._out_shape = image.shape
        return image

    def valid_char(self, char, same_line=False):
        """Check if char is a valid character to include in image.

        Args:
            char (Character): The character to validate.
            same_line (bool): Consider whether char is in the same line
                as those already in the image example.

        Returns:
            bool: True for valid, False otherwise.

        """
        valid = True
        if same_line and len(self._x_raw) > 0:
            if not ops.in_line(xmin_line=self._x_raw,
                               xmax_line=self._char_xmax,
                               ymin_line=min(self._y_raw),
                               xmin_new=char.x,
                               xmax_new=char.x + char.w,
                               ymax_new=char.y + char.h):
                valid = False
        if char.image_id not in self.filepath:
            # not in same image
            valid = False
        return valid

    def add_char(self, char):
        """Add a character to the image.

        Args:
            char (Character): The character to add.

        """
        if self.valid_char(char):
            self._x_raw.append(char.x)
            self._y_raw.append(char.y)
            self._w_raw.append(char.w)
            self._h_raw.append(char.h)
            self._labels.append(char.label)
            self._blocks.append(char.block_id)
            self._ids.append(char.id)
        else:
            raise ValueError(
                "Invalid image id '{}'.".format(char.image_id),
                "Must be within {}.".format(self.filepath)
            )

    def combine_with(self, images):
        """

        Args:
            images (list of ImageMeta):

        """
        raise NotImplementedError

    def _load_image(self):
        """Loads the raw parent image."""

        with tf.gfile.FastGFile(self.filepath, 'rb') as f:

            encoded_image = f.read()

            try:
                image = self._image_helper.decode_jpeg(encoded_image)
            except (tf.errors.InvalidArgumentError, AssertionError):
                print("Skipping file with invalid JPEG data: %s" % image_path)
                return

            return image

    @property
    def _char_xmax(self):
        """Maximum x-coordinate (column) of each character."""
        return [self._x_raw[i] + self._w_raw[i] for i in range(self.num_chars)]

    @property
    def _char_ymax(self):
        """Maximum y-coordinate (row) of each character."""
        return [self._y_raw[i] + self._h_raw[i] for i in range(self.num_chars)]

    def new_shape(self, shape, ratio=False):
        """Resolves (and computes) input shape to a consistent type.

        Args:
            shape (tuple or None): New shape of image (height, width),
                with potentially inconsistent types.
            ratio (bool): Return new size as ratio of original size.

        Returns:
            int or float: Absolute or relative height
            int or float: Absolute or relative width

        """
        height = self.height
        width = self.width

        if shape and any(shape):
            assert all([not a or isinstance(a, (int, float))
                        for a in shape]), "Invalid shape {}".format(shape)

            if isinstance(shape[0], int):
                height = shape[0]
            elif isinstance(shape[0], float):
                height = height * shape[0]

            if isinstance(shape[1], int):
                width = shape[1]
            elif isinstance(shape[1], float):
                width = width * shape[1]

            # Compute to maintain aspect ratio
            if not shape[0]:
                height = height * (width / self.width)
            if not shape[1]:
                width = width * (height / self.height)

        if ratio:
            height = height / self.height
            width = width / self.width
        else:
            height = int(height)
            width = int(width)

        return height, width
