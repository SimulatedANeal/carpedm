#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
#
# Portions of this module are taken or adapted from the TensorFlow
# CIFAR-10 estimator tutorial, so here is their license.
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


"""Training utilities.

This modules provides utilities for training machine learning models.
It uses or makes slight modifications to code from the
`TensorFlow CIFAR-10 estimator tutorial`_.

..  _TensorFlow CIFAR-10 estimator tutorial:
    https://github.com/tensorflow/models/tree/master/tutorials/image

"""
import itertools

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter


def config_optimizer(params):
    """Configure the optimizer used for training.

    Sets the learning rate schedule and optimization algorithm.

    Args:
        params (tf.contrib.training.HParams): Hyperparameters.

    Returns:
        tf.train.Optimizer
    """
    # Learning rate schedule
    if params.lr_decay_steps:
        learning_rate = tf.train.exponential_decay(
            params.learning_rate,
            tf.train.get_global_step(),
            params.lr_decay_steps,
            params.lr_decay_rate,
            staircase=True
        )
        tf.summary.scalar("learning_rate", learning_rate)
    # elif params.staged_lr:
    #     # tf.train.piecewise_constant(tf.train.get_global_step(),
    #     #                             boundaries=None,
    #     #                             staged_lr=params.staged_lr)
    #     raise NotImplementedError
    else:
        learning_rate = params.learning_rate

    # Optimizer
    if params.optimizer == 'sgd' and params.momentum:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=params.momentum)
    elif params.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif params.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif params.optimizer == 'rmsprop' and params.momentum:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              momentum=params.momentum)
    elif params.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

    return optimizer


# ========= BEGIN Adapted from TensorFlow CIFAR 10 tutorial ========= #

class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second.
      Total time is tracked and then divided by the total number of steps
      to get the average step time and then batch_size is used to determine
      the running average of examples per second. The examples per second for the
      most recent interval is also logged.
    """

    def __init__(
            self,
            batch_size,
            every_n_steps=10,
            every_n_secs=None,):
        """Initializer for ExamplesPerSecondHook.
          Args:
          batch_size: Total batch size used to calculate examples/second from
          global time.
          every_n_steps: Log stats every n steps.
          every_n_secs: Log stats every n seconds.
        """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                             average_examples_per_sec, current_examples_per_sec,
                             self._total_steps)


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()
    return _local_device_chooser


def group_train_op(optimizer, gradvars, update_ops):
    train_op = [
        optimizer.apply_gradients(
            gradvars, global_step=tf.train.get_global_step())
    ]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)
    return train_op


def make_hooks(tensors_to_log, optimizer, num_workers, params):
    examples_sec_hook = ExamplesPerSecondHook(
        params.train_batch_size, every_n_steps=100)
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    train_hooks = [logging_hook, examples_sec_hook]
    if params.sync:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

    return train_hooks


def compute_global_grads_loss(tower_gradvars, tower_losses):
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in all_grads.items():
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))

    loss = tf.reduce_mean(tower_losses, name='loss')
    return gradvars, loss
