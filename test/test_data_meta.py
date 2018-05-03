#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

import unittest

from carpedm.data.small import path as sample
from carpedm.data.meta import MetaLoader


class MetaLoaderTestCase(unittest.TestCase):

    N_CHARS = {
        '200003076': 230,
        '200003967': 176,
        '200014740': 293,
        '200021637': 208,
        '200021660': 71,
        '200021712': 65,
        '200021763': 40,
        '200021802': 177,
        '200021851': 101,
        '200021853': 85,
        '200021869': 153,
        '200021925': 157,
        '200022050': 516,
        'brsk00000': 211,
        'hnsd00000': 148
    }

    def test_invalid_split(self):

        with self.assertRaisesRegex(AssertionError, 'Invalid ID'):
            MetaLoader(data_dir=sample, test_split='blah')

        with self.assertRaisesRegex(AssertionError, 'Invalid split'):
            MetaLoader(data_dir=sample, test_split=1.3)
            MetaLoader(data_dir=sample, dev_split='23.8')
            MetaLoader(data_dir=sample, dev_split=-0.7)

        with self.assertRaisesRegex(ValueError, 'Invalid split'):
            MetaLoader(data_dir=sample, test_split=1)
            MetaLoader(data_dir=sample, dev_split='hnsd00000')

    def test_split_str4test_str4dev(self):
        tsplit = 'hnsd00000'
        dsplit = ['200021712', '200021763']
        num_dev = sum(map(lambda x: self.N_CHARS[x], dsplit))
        pmjt = MetaLoader(data_dir=sample, test_split=tsplit,
                          dev_split=','.join(dsplit), image_scope='char')
        self.assertEqual(self.N_CHARS[tsplit], len(pmjt._image_meta['test']))
        self.assertEqual(num_dev, len(pmjt._image_meta['dev']))

    def test_split_str4test_float4dev(self):
        tsplit = '200021925'
        dsplit = 0.2
        num_dev = int(sum(self.N_CHARS.values()) * dsplit)
        pmjt = MetaLoader(data_dir=sample, test_split=tsplit,
                          dev_split=dsplit, image_scope='char')
        self.assertEqual(self.N_CHARS[tsplit], len(pmjt._image_meta['test']))
        self.assertEqual(num_dev, len(pmjt._image_meta['dev']))

    def test_split_float4test_str4dev(self):
        tsplit = 0.153
        dsplit = 'brsk00000'
        num_test = int(sum(self.N_CHARS.values()) * tsplit)
        pmjt = MetaLoader(data_dir=sample, test_split=tsplit,
                          dev_split=dsplit, image_scope='char')
        self.assertEqual(num_test, len(pmjt._image_meta['test']))
        self.assertEqual(self.N_CHARS[dsplit], len(pmjt._image_meta['dev']))

    def test_split_float4test_float4dev(self):
        tsplit = '0.3'
        dsplit = 0.5
        total = sum(self.N_CHARS.values())
        num_test = int(total * float(tsplit))
        num_dev = int(total * dsplit)
        pmjt = MetaLoader(data_dir=sample, test_split=tsplit,
                          dev_split=dsplit, image_scope='char')
        self.assertEqual(num_test, len(pmjt._image_meta['test']))
        self.assertEqual(num_dev, len(pmjt._image_meta['dev']))


if __name__ == '__main__':
    unittest.main()
