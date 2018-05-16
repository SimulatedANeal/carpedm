#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""This module provides methods for summarizing dataset statistics."""
from collections import Counter

from carpedm.data.util import ImageMeta


def ratio(class_label, data):
    return float(data.count(class_label)) / len(data)


def majority(tokens):
    counts = Counter(tokens)
    major, count = counts.most_common(1)[0]
    r = ratio(major, tokens)
    return major, count, r

