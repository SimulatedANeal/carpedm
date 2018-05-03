#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Data exploration.

If this file is changed, please also change the ``:lines:`` option in
the following files where this code is referenced with the
``literalinclude`` directive.

    * carpedm/data/meta.py
    * ../guides/usage.rst

"""
import carpedm as dm


# Create objects for storing meta data
single_kana = dm.data.MetaLoader(data_dir=dm.data.SAMPLE, image_scope='char', charset='kana')
kanji_seq = dm.data.MetaLoader(data_dir=dm.data.SAMPLE, image_scope='seq', seq_len=3, charset='kanji')
full_page = dm.data.MetaLoader(data_dir=dm.data.SAMPLE, image_scope='page', charset='all')

# View images
single_kana.view_images(subset='train', shape=(64,64))
kanji_seq.view_images(subset='dev', shape=(None, 64))
full_page.view_images(subset='test', shape=None)

# Save the data as TFRecords (default format_store)
single_kana.generate_dataset(out_dir='/tmp/pmjtc_data', subset='train')
