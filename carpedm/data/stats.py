#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""This module provides methods for summarizing dataset statistics."""
import os
import sys
import warnings
from collections import Counter

import numpy as np

from carpedm.data.lang import code2char


def ratio(class_label, data):
    """Compute ratio of all data that is `class_label`."""
    try:
        return float(data.count(class_label)) / len(data)
    except ZeroDivisionError:
        warnings.warn("data {} is empty".format(id(data)))
        return 0.


def majority(tokens):
    """Compute majority class.

    Args:
        tokens (list): The data

    Returns:
        (str, int, float): majority class, count, ratio

    """
    counts = Counter(tokens)
    if len(counts) > 0:
        major, count = counts.most_common(1)[0]
    else:
        major, count = None, 0.
    r = ratio(major, tokens)
    return major, count, r


class ClassCounts(object):
    """Class for storing and plotting or printing class counts."""

    def __init__(self):
        self._data = dict()
        self._colors = ['green', 'blue', 'red']

    def add_dataset(self, data, label, color=None):
        """Add dataset to class' dictionary.

        Args:
            data (list): Flat list of data classes to count.
            label (str): ID for the dataset.
            color (str or None): Color to use for this data if plotting.

        Returns:
            nothing

        """
        self._data[label] = (data, color)

    def plot_counts(self, vocab, include, figsize=(10, 8), save_dir=None):
        """Plot counts in histogram if plotting installed, else print.

        Args:
            vocab (carpedm.data.lang.Vocabulary):
            n_include (tuple): Include class IDs (in vocab) in this range.
            figsize (tuple): Figure size.
            save_dir (str): Save figures/files to this directory.

        Returns:
            nothing

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Plotting is not available."
                          "Please install matplotlib if you wish to use it.")
            self.print_counts(save_dir=save_dir)
        else:
            from carpedm.data.viewer import font

            # rr = vocab.get_num_reserved()
            start = vocab.get_num_reserved() if not include[0] else include[0]
            stop = vocab.get_num_classes() if not include[1] else include[1]
            # center groups of bars
            nn = stop - start + 1
            bar_width = 1.0 / (len(self._data) + 1)
            ind = np.arange(nn)
            centers = ind + bar_width * len(self._data) / 2.

            fig, ax = plt.subplots(figsize=figsize)
            max_count = 0
            rects = []

            for i, (data, color) in enumerate(self._data.values()):
                ids = [vocab.char_to_id(c) for c in data
                       if start <= vocab.char_to_id(c) <= stop]
                counts = Counter(ids)
                mc = max(counts.values())
                max_count = mc if mc > max_count else max_count
                x = ind + i * bar_width
                y = [0] * nn
                for cid, count in counts.items():
                    y[cid - start] = count
                rects.append(ax.bar(x, y, bar_width,
                                    color=color or self._colors[i]))
            char_list = [vocab.id_to_char(i) for i in range(start, stop + 1)]
            char_list = map(code2char, char_list)

            ax.set_ylabel("Counts")
            ax.set_ylim([0, max_count])
            ax.set_xlim([0, nn + 1])
            ax.set_title("Characters {0}-{1} Relative Frequency".format(
                start, stop))
            ax.set_xlabel("Characters")
            ax.set_xticks(centers)
            ax.set_xticklabels(char_list, fontproperties=font(10))
            ax.legend([r[0] for r in rects], self._data.keys())
            plt.tight_layout()
            if save_dir:
                filename = os.path.join(
                    save_dir, "{0}-{1}_frequency.svg".format(start, stop))
                plt.savefig(filename)
            plt.show()


    def print_counts(self, save_dir=None):
        """Print frequenci

        Args:
            save_dir (str or None): Directory to which count files are
                saved.

        Returns:
            nothing

        """
        for did in self._data:
            fp = open(os.path.join(save_dir, did + '_counts.csv'), 'w') \
                if save_dir else sys.stdout
            # Header
            if not save_dir:
                print("{} Token Counts\n{}".format(did, '-' * 22))
            else:
                print('class,(unicode),count', file=fp)

            counts = Counter(self._data[did][0])
            for t in counts.most_common():
                print("{0},({1}),{2}".format(t[0], code2char(t[0]), t[1]),
                      file=fp)
            if save_dir:
                fp.close()
