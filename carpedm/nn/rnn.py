#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Recurrent layers and components."""

import tensorflow as tf


def bi_lstm(x, n_layers, n_units):

    lstm_fw_cells = [tf.nn.rnn_cell.BasicLSTMCell(n_units)
                     for i in range(n_layers)]
    lstm_bw_cells = [tf.nn.rnn_cell.BasicLSTMCell(n_units)
                     for i in range(n_layers)]
    batch_size = tf.shape(x)[0]

    for i in range(n_layers):
        cell_fw = lstm_fw_cells[i]
        cell_bw = lstm_bw_cells[i]
        state_fw = cell_fw.zero_state(batch_size, tf.float32)
        state_bw = cell_bw.zero_state(batch_size, tf.float32)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x,
            initial_state_fw=state_fw,
            initial_state_bw=state_bw,
            scope='BLSTM_'+str(i),
            dtype=tf.float32
        )
        x = tf.concat([output_fw, output_bw], 2)

    return x
