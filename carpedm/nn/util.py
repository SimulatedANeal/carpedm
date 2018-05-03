#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
# Portions of this module are lightly modified or taken directly from
# the TensorFlow CIFAR-10 image tutorial, so here is their license.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Utilities for managing and visualizing neural network layers."""
import re

import tensorflow as tf


TOWER_NAME = 'tower'


def name_nice(raw):
    """Convert tensor name to a nice format.

    Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
    training session. This helps the clarity of presentation on
    tensorboard.
    """
    return re.sub(r'(%s_[0-9]*/)' % TOWER_NAME, '', raw)


def activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    name = name_nice(x.op.name)
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
