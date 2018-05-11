#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Evaluation helpers."""

import numpy as np
import tensorflow as tf


def confusion_matrix_metric(labels, predictions, num_classes):
    """

    Args:
        self:
        labels:
        predictions:

    Returns:

    """
    confusion = tf.get_variable(name='confusion',
                                shape=[num_classes, num_classes],
                                dtype=tf.int32,
                                initializer=tf.initializers.zeros,
                                trainable=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])

    confusion_batch = tf.confusion_matrix(
        tf.reshape(labels, [-1]),
        tf.reshape(predictions, [-1]),
        num_classes
    )

    confusion_update = tf.assign_add(confusion, confusion_batch)

    return tf.convert_to_tensor(confusion), tf.group(confusion_update)


def plot_confusion_matrix(cm, classes,
                          normalize=False, save_as=None,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    from carpedm.data import font

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties=font(8))
    plt.yticks(tick_marks, classes, fontproperties=font(8))

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_as:
        plt.savefig(save_as)
    plt.show()
