#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

import csv
import unittest

from matplotlib.pyplot import imread

from carpedm.data.small import path as sample
from carpedm.data.util import Character, ImageMeta


class ImageMetaTestCase(unittest.TestCase):

    csv_path = sample + '/200021660/200021660_coordinate.csv'
    image_path = sample + '/200021660/images/200021660-00005_1.jpg'
    char1 = Character(label='U+4E4B', image_id=image_path, x=2314, y=717,
                      block_id='B0001', char_id='C0001', w=179, h=132)
    char2 = Character(label='U+6CD5', image_id=image_path, x=2322, y=963,
                      block_id='B0001', char_id='C0002', w=149, h=168)

    def test_full_width(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        self.assertEqual(meta.full_w, 2834)

    def test_full_height(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        self.assertEqual(meta.full_h, 4240)

    def test_get_multi_char_xmin(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.xmin, 2314)

    def test_get_multi_char_ymin(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.ymin, 717)

    def test_get_multi_char_xmax(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.xmax, 2314 + 179)

    def test_get_multi_char_ymax(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.ymax, 963 + 168)

    def test_get_multi_char_width(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.width, 179)

    def test_get_multi_char_height(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.height, 414)

    def test_num_characters(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.num_chars, 2)

    def test_character_labels(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        self.assertEqual(meta.char_labels, ['U+4E4B', 'U+6CD5'])

        # TODO: test for adding character out of reading order after functionality implemented

    def test_character_bounding_boxes(self):
        meta = ImageMeta(filepath=self.char1.image_id, full_image=False,
                         first_char=self.char1)
        meta.add_char(self.char2)
        # char1
        self.assertEqual(meta.char_bboxes[0].xmin, 0)
        self.assertEqual(meta.char_bboxes[0].xmax, 179)
        self.assertEqual(meta.char_bboxes[0].ymin, 0)
        self.assertEqual(meta.char_bboxes[0].ymax, 132)
        # char2
        self.assertEqual(meta.char_bboxes[1].xmin, 8)
        self.assertEqual(meta.char_bboxes[1].xmax, 157)
        self.assertEqual(meta.char_bboxes[1].ymin, 246)
        self.assertEqual(meta.char_bboxes[1].ymax, 414)

    def test_line_bounding_boxes_for_full_page(self):
        meta = ImageMeta(filepath=self.image_path, full_image=True)
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # header
            for row in reader:
                meta.add_char(Character(*row))
        # line1
        self.assertEqual(meta.line_bboxes[0].xmin, 2290)
        self.assertEqual(meta.line_bboxes[0].xmax, 2493)
        self.assertEqual(meta.line_bboxes[0].ymin, 717)
        self.assertEqual(meta.line_bboxes[0].ymax, 3780)
        # line2
        self.assertEqual(meta.line_bboxes[1].xmin, 1931)
        self.assertEqual(meta.line_bboxes[1].xmax, 2136)
        self.assertEqual(meta.line_bboxes[1].ymin, 703)
        self.assertEqual(meta.line_bboxes[1].ymax, 3796)
        # line3
        self.assertEqual(meta.line_bboxes[2].xmin, 1606)
        self.assertEqual(meta.line_bboxes[2].xmax, 1811)
        self.assertEqual(meta.line_bboxes[2].ymin, 715)
        self.assertEqual(meta.line_bboxes[2].ymax, 3783)
        # line4
        self.assertEqual(meta.line_bboxes[3].xmin, 1266)
        self.assertEqual(meta.line_bboxes[3].xmax, 1485)
        self.assertEqual(meta.line_bboxes[3].ymin, 701)
        self.assertEqual(meta.line_bboxes[3].ymax, 3320)
        # line5
        self.assertEqual(meta.line_bboxes[4].xmin, 940)
        self.assertEqual(meta.line_bboxes[4].xmax, 1154)
        self.assertEqual(meta.line_bboxes[4].ymin, 859)
        self.assertEqual(meta.line_bboxes[4].ymax, 2368)
        # line6
        self.assertEqual(meta.line_bboxes[5].xmin, 612)
        self.assertEqual(meta.line_bboxes[5].xmax, 817)
        self.assertEqual(meta.line_bboxes[5].ymin, 1627)
        self.assertEqual(meta.line_bboxes[5].ymax, 3583)

    def test_line_bounding_boxes_for_not_full_page(self):
        meta = ImageMeta(filepath=self.image_path, full_image=False,
                         first_char=self.char1)
        self.assertEqual(meta.line_bboxes, [])

    def test_load_image(self):
        """

        Note that pixel values may differ slightly due to the IFAST
        method TensorFlow uses for decoding images.

        """
        ref = imread(self.image_path)
        meta = ImageMeta(self.image_path, full_image=False)

        my_image = meta.load_image(None)
        self.assertEqual(ref.shape, my_image.shape)

        my_image = meta.load_image((None, None))
        self.assertEqual(ref.shape, my_image.shape)

        my_image = meta.load_image((0.5, 0.5))
        shape = (ref.shape[0] / 2, ref.shape[1] / 2, 3)
        self.assertEqual(shape, my_image.shape)

        shape = (1000, 500)
        my_image = meta.load_image(shape)
        self.assertEqual(shape + (3,), my_image.shape)

        my_image = meta.load_image((0.3, 400))
        shape = (ref.shape[0] * 0.3, 400)
        self.assertEqual(shape + (3,), my_image.shape)

        my_image = meta.load_image((0.74, None))
        shape = (int(ref.shape[0] * 0.74), int(ref.shape[1] * 0.74))
        self.assertEqual(shape + (3,), my_image.shape)

    def test_valid_character(self):
        # NOTE: Does not handle whether character is in char_set.
        meta = ImageMeta(filepath=self.image_path, full_image=False,
                         first_char=self.char1)
        self.assertFalse(meta.valid_char(Character(label='U+34cf',
                                                   image_id='wrong',
                                                   x=0, y=0, block_id='B0100',
                                                   char_id='C0001', w=0, h=0)))
        self.assertTrue(meta.valid_char(self.char2, same_line=True))
        other_line_char = Character('U+738B', self.image_path, x=1652, y=2092,
                                    block_id='B0001', char_id='C0035',
                                    w=119, h=115)
        self.assertTrue(meta.valid_char(other_line_char))
        self.assertFalse(meta.valid_char(other_line_char, same_line=True))


if __name__ == '__main__':
    unittest.main()
