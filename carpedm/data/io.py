#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
# The DataWriter helper class of this module is based on the TensorFlow
# "im2txt" models input pipeline, so here is their license:
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

"""Input and output.

This module provides functionality for reading and writing data.

Todo:
    * Tests
        * DataWriter
        * CSVParser
"""
import csv
import functools
import os
import random
import sys
from datetime import datetime
from threading import Thread

import numpy as np

from carpedm.data import ops
from carpedm.data.util import Character, ImageMeta
from carpedm.data.util import image_path


class DataWriter(object):
    """Utility for writing data to disk in various formats.

    Attributes:
        available_formats (list): The available formats.

    References:
        Heavy modification of ``_process_dataset`` in the
        `input pipeline`_ for the TensorFlow `im2txt` models.

    ..  _input pipeline: https://github.com/tensorflow/models/blob/
        master/research/im2txt/im2txt/data/build_mscoco_data.py

    """

    available_formats = ['tfrecords', 'jpeg', 'jpg', 'png']

    def __init__(self, format_out, images, image_shape, vocab,
                 chunk, character, line, label, bbox, subdirs):
        """

        Args:
            format_out (str):
            images (list of ImageMeta):
            image_shape (tuple or None):
            vocab (Vocabulary):
            chunk (bool):
            character (bool):
            line (bool):
            label (bool):
            bbox (str or None): If not None, include bbox in features
                as unit (e.g. 'pixel', 'ratio' [of image]))
            subdirs (bool): Generate a subdirectory for each class.
        """

        self._writer_types = {
            'tfrecords': self._write_tfrecords,
            'jpeg': functools.partial(self._write_raw_images,
                                      image_format='.jpeg'),
            'jpg': functools.partial(self._write_raw_images,
                                     image_format='.jpg'),
            'png': functools.partial(self._write_raw_images,
                                     image_format='.png')
        }

        assert format_out in self.available_formats
        self._write = self._writer_types[format_out]
        self._images = images
        self._shape = image_shape
        self._vocab = vocab
        self._chunk = chunk
        self._char = character
        self._line = line
        self._label = label
        self._bbox = bbox
        self._subdirs = subdirs

    def write(self, fname_prefix, num_threads, num_shards):
        """Write data to disk.

        Args:
            fname_prefix (str): Path base for data files.
            num_threads (int): Number of threads to run in parallel.
            num_shards (int): Total number of shards to write, if any.

        Returns:
            int: Total number of examples written.

        """
        spacing = np.linspace(0, len(self._images),
                              num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        num_examples = [0] * num_threads
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        print("Launching %d threads for spacings: %s" % (
            num_threads, ranges
        ))
        for thread_ix in range(len(ranges)):
            args = (thread_ix, num_shards, fname_prefix, ranges, num_examples)
            t = Thread(target=self._write, args=args)
            t.start()
            threads.append(t)

        # Try joining threads with TensorFlow
        try:
            import tensorflow as tf
        except ImportError:
            for thread in threads:
                thread.join()
        else:
            coord = tf.train.Coordinator()
            coord.join(threads)

        print("%s: Finished writing all %d image-target pairs." %
              (datetime.now(), len(self._images)))
        return sum(num_examples)

    def _write_tfrecords(self, index, num_shards,
                         fname_prefix, ranges, num_examples):
        """Write TFRecords.

        Args:
            index (int): Thread identifier in [0, len(ranges)].
            ranges (list): Pairs of integers specifying the ranges of
                the dataset to process in parallel
            num_examples (list): Structure for storing number of examples
                written.

        Returns:
            int: Number of examples written to shard.

        """
        try:
            import tensorflow as tf
        except ImportError:
            import warnings
            warnings.warn("The tfrecords output format is not available. "
                          "Please install tensorflow if you wish to use it.")
            return

        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)
        shard_ranges = np.linspace(ranges[index][0],
                                   ranges[index][1],
                                   num_shards_per_batch + 1).astype(int)
        written_count = 0
        for s in range(num_shards_per_batch):
            shard = index * num_shards_per_batch + s
            output_file = "%s-%.5d-of-%.5d" % (fname_prefix, shard+1,
                                               num_shards)
            images_in_shard = np.arange(shard_ranges[s],
                                        shard_ranges[s + 1],
                                        dtype=int)
            images_in_shard = [self._images[i] for i in images_in_shard]

            shard_counter = 0
            writer = tf.python_io.TFRecordWriter(output_file)
            for img_meta in images_in_shard:
                feature_dicts = img_meta.generate_features(
                    image_shape=self._shape, chunk=self._chunk,
                    character=self._char, vocab=self._vocab,
                    line=self._line, label=self._label, bbox=self._bbox)
                for fd in feature_dicts:
                    example = ops.to_sequence_example(fd)
                    if example is not None:
                        writer.write(example.SerializeToString())
                        written_count += 1
                        shard_counter += 1

                if not shard_counter % 1000:
                    print("%s [thread %d]: Processed %d items in thread batch."
                          % (datetime.now(), index, shard_counter))
                    sys.stdout.flush()

            writer.close()
            print("%s [thread %d]: Wrote %d image-target pairs to %s" %
                  (datetime.now(), index, shard_counter, output_file))
            sys.stdout.flush()

        num_examples[index] = written_count

    def _write_raw_images(self, index, num_shards, fname_prefix, ranges,
                          num_examples, image_format):
        """Write structured directories of images."""
        from PIL import Image

        os.makedirs(fname_prefix, exist_ok=True)

        def create_csv(name, fields):
            csvfile = open(
                os.path.join(fname_prefix, '{}_{}.csv'.format(name, index)),
                'w+')
            writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
            writer.writeheader()
            return csvfile, writer

        tfile, twriter = create_csv(
            name='targets', fields=['file_id',
                                    'unicodes',
                                    'bounding_boxes (x:y:h:w)',]
        )

        pfile, pwriter = create_csv(
            name='parents', fields=['file_id',
                                    'image_id',
                                    'X',
                                    'Y',
                                    'Width',
                                    'Height']
        )

        def write_features(meta, feats, count):
            target_dict = dict()
            parent_dict = dict()

            im = Image.fromarray(feats['image/data'])
            im_id = '{}_{}'.format(index, count)
            target_dict['file_id'] = im_id
            parent_dict['file_id'] = im_id

            # Parent meta data
            parent_dict['image_id'] = meta.filepath.split('/')[-1].strip('.jpg')
            parent_dict['X'] = meta.xmin
            parent_dict['Y'] = meta.ymin
            parent_dict['Width'] = meta.width
            parent_dict['Height'] = meta.height

            if self._char and self._label:
                label = "_".join([self._vocab.id_to_char(c)
                                  for c in feats['image/seq/char/id']])
                if self._subdirs:
                    dir_img = os.path.join(fname_prefix, label)
                    os.makedirs(dir_img, exist_ok=True)
                    fname = os.path.join(dir_img, im_id + image_format)
                else:
                    fname = os.path.join(fname_prefix, im_id + image_format)
                target_dict['unicodes'] = label
            else:
                fname = os.path.join(fname_prefix, im_id + image_format)

            if self._char and self._bbox:
                xmin = feats['image/seq/char/bbox/xmin']
                ymin = feats['image/seq/char/bbox/ymin']
                xmax = feats['image/seq/char/bbox/xmax']
                ymax = feats['image/seq/char/bbox/ymax']

                target_dict['bounding_boxes (x:y:h:w)'] = "_".join(
                    ["{}:{}:{}:{}".format(xmin[i],
                                          ymin[i],
                                          xmax[i] - xmin[i],
                                          ymax[i] - ymin[i])
                     for i in range(feats['image/char/count'])]
                )

            im.save(fname)
            twriter.writerow(target_dict)
            pwriter.writerow(parent_dict)

        written_count = 0
        first, last = ranges[index]
        images = [self._images[i] for i in range(first, last)]

        for img in images:
            feature_dicts = img.generate_features(
                image_shape=self._shape, chunk=self._chunk,
                character=self._char, vocab=self._vocab,
                line=self._line, label=self._label, bbox=self._bbox)
            for fd in feature_dicts:
                write_features(meta=img, feats=fd, count=written_count)
                written_count += 1

            if not written_count % 1000:
                print("%s [thread %d]: Processed %d items in thread batch."
                      % (datetime.now(), index, written_count))
                sys.stdout.flush()

        num_examples[index] = written_count

        tfile.close()
        pfile.close()


class CSVParser(object):
    """Utility class for parsing coordinate CSV files."""

    x = 'X'
    y = 'Y'
    w = 'Width'
    h = 'Height'
    image = 'Image'
    label = 'Unicode'
    bid = 'Block ID'
    cid = 'Char ID'

    def __init__(self, csv_file, data_dir, bib_id, ):
        """Initializer.

        Args:
            csv_file (file): Opened CSV file to parse.
            data_dir (str): Path to directory containing raw data.
            bib_id (str): Bibliography ID.
        """
        self._reader = csv.DictReader(csv_file, restkey='comments')
        self._data_dir = data_dir
        self._bib_id = bib_id

    def character(self, row):
        """Convert CSV row to a Character object.

        Returns:
            Character: The next character
        """
        row = {k.strip('\ufeff'): v for k, v in row.items()}

        return Character(
            x=row[self.x], y=row[self.y], w=row[self.w], h=row[self.h],
            label=row[self.label], block_id=row[self.bid],
            char_id=row[self.cid], image_id=image_path(self._data_dir,
                                                       self._bib_id,
                                                       row[self.image])
        )

    def characters(self):
        """Generates rest of characters in CSV.

        Yields:
            :obj:`carpedm.data.util.Character`: The next character.
        """
        for row in self._reader:
            yield self.character(row)

    def parse_pages(self):
        """Genereate metadata for full page images.

        Includes every character on page. Characters not in character
        set or vocabulary will be labeled as unknown when converted to
        integer IDs.

        Returns:
            :obj:`list` of :obj:`carpedm.data.util.ImageMeta`:
                Page image meta data.
        """
        pages = []
        char = self.character(next(self._reader))
        image_id = char.image_id
        image = ImageMeta(
            filepath=image_id, full_image=True, first_char=char)
        for char in self.characters():
            if char.image_id == image_id:
                image.add_char(char)
            else:
                pages.append(image)
                image_id = char.image_id
                image = ImageMeta(
                    filepath=image_id, full_image=True, first_char=char)
        # Add last image.
        pages.append(image)
        return pages

    def parse_lines(self):
        """Generate metadata for vertical lines of characters.

        Characters not in character set or vocabulary will be labeled as
        unknown when converted to integer IDs.

        Returns:
            :obj:`list` of :obj:`carpedm.data.util.ImageMeta`:
                Line image meta data.
        """
        lines = []

        c = self.character(next(self._reader))
        image = ImageMeta(filepath=c.image_id, first_char=c)

        for c in self.characters():
            if image.valid_char(c, same_line=True):
                image.add_char(c)
            else:
                lines.append(image)
                image = ImageMeta(filepath=c.image_id, first_char=c)
        lines.append(image)
        return lines

    def parse_sequences(self, charset, len_min, len_max):
        """Generate metadata for images of character sequences.

        Only includes sequences of chars in the desired character set.
        If ``len_min == len_max``, sequence length is deterministic, else
        each sequence is of random length from [len_min, len_max].

        Args:
            charset (CharacterSet): The character set.
            len_min (int): Minimum sequence length.
            len_max (int): Maximum sequence length.

        Returns:
            :obj:`list` of :obj:`carpedm.data.util.ImageMeta`:
                Sequence image meta data.

        """
        sequences = []
        length = random.randint(len_min, len_max)
        image = None
        for c in self.characters():
            if image is None:
                if charset.in_charset(c.label):
                    image = ImageMeta(filepath=c.image_id, first_char=c)
            elif (image.valid_char(c, same_line=True)
                  and charset.in_charset(c.label)
                  and image.num_chars < length):
                image.add_char(c)
            else:
                if len_min <= image.num_chars <= len_max:
                    sequences.append(image)
                    length = random.randint(len_min, len_max)
                if charset.in_charset(c.label):
                    image = ImageMeta(filepath=c.image_id, first_char=c)
                else:
                    image = None
        if image is not None and len_min <= image.num_chars <= len_max:
            sequences.append(image)
        return sequences

    def parse_characters(self, charset):
        """Generate metadata for single character images.

        Args:
            charset (CharacterSet): Character set.

        A more efficient implementation of ``parse_sequences`` when
        ``image_scope='seq'`` and ``seq_len=1``.

        Only characters in the character set are included.

        Returns:
            :obj:`list` of :obj:`carpedm.data.util.ImageMeta`:
                Single character image meta data.
        """
        return [ImageMeta(filepath=c.image_id, first_char=c)
                for c in self.characters() if charset.in_charset(c.label)]
