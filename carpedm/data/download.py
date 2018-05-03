# -*- coding: utf-8 -*-
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.


"""Download scripts.

This module provides the interface for downloading raw datasets from
their source.

.. include:: ../../../DESCRIPTION.rst
    :start-after: machine learning researchers.
    :end-before: Though still in the early stages

Example:
    Data may be downloaded externally using the provided script:

    .. code-block:: bash

        $ download_data --data-dir <download/to/this/directory> --data-id pmjtc

.. note::

    If an expected data subdirectory already exists in the
    specified target ``data-dir`` that data will not be downloaded, even
    if the subdirectory is empty. This should be fixed in a future
    version.

Todo:

    * Update ``get_books_list`` once list is included in downloadables.
    * Check subdirectory contents.
    * Generalize download structure for other datasets.

..  _Dataset of Pre-Modern Japanese Text Character Shapes:
    http://codh.rois.ac.jp/char-shape/
"""
import argparse
import os
import zipfile


# Base URLS for datasets
_URLS = {
    'pmjtc': "http://codh.rois.ac.jp/char-shape/book/",
}
_BOOKS = {
    'pmjtc': [
        # "bib_id",  # num_page_images, types, tokens
        "200003076",  # 346, 1720, 63959
        "200003967",  # 88, 1119, 11197
        "200014740",  # 182, 1969, 44832
        "200021637",  # 37,  417,  4871
        "200021660",  # 185, 1758, 32525
        "200021712",  # 165,  843, 24480
        "200021763",  # 100,  704, 11397
        "200021802",  # 111,  560, 19575
        "200021851",  # 59,  430,  5599
        "200021853",  # 79,  595,  9046
        "200021869",  # 35,  330,  3003
        "200021925",  # 45,  693,  4259
        "200022050",  # 30,  255,  9545
        "brsk00000",  # 238, 2197, 75462
        "hnsd00000",  # 522, 1972, 83492
    ]
}


def get_books_list(dataset='pmjtc'):
    """Retrieve list of books/images in dataset.

    Args:
        dataset (str): Identifier for dataset for which to retrieve
            information.

    Returns:
        :obj:`list` of :obj:`str`: Names of dataset subdirectories
        and/or files.
    """
    return _BOOKS[dataset]


def maybe_download(directory, dataset='pmjtc'):
    """Download character dataset if BOOKS not in directory.

    Args:
        directory (str): Directory where dataset is located or
            should be saved.
        dataset (str): Identifier for dataset to download.

    """
    from urllib.request import urlretrieve

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for bib_id in get_books_list():
        if not os.path.exists(os.path.join(directory, bib_id)):
            print("Could not find %s in %s" % (bib_id, directory))
            filename = bib_id + '.zip'
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filename):
                url = os.path.join(_URLS[dataset], bib_id, filename)
                print("Downloading %s to %s" % (url, filepath))
                inprogress_filepath = filepath + ".incomplete"
                inprogress_filepath, _ = urlretrieve(url, inprogress_filepath)

                os.rename(inprogress_filepath, filepath)
                statinfo = os.stat(filepath)
                print("Successfully downloaded %s, %s bytes." %
                      (filename, statinfo.st_size))

            unzip_dir = os.path.join(directory, filename.strip(".zip"))
            if not os.path.exists(unzip_dir):
                print("Unzipping files...", end='', flush=True)
                zipfile.ZipFile(filepath, "r").extractall(directory)
                print("done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help="""\
                        Directory to which dataset books will be 
                        downloaded if not already present.\
                        """)
    parser.add_argument('-i', '--data-id', type=str, default='pmjtc',
                        choices=['pmjtc'],
                        help="Identifier for dataset to download.")
    args = parser.parse_args()
    maybe_download(args.data_dir, args.data_id)
