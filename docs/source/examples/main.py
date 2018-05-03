#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Minimal main module.

If this file is changed, please also change the ``:lines:`` option in
the following files where this code is referenced with the
``literalinclude`` directive.

    * ../guides/usage.rst

"""
import os
import re

import tensorflow as tf

import carpedm as dm
from carpedm.util import registry


tf.logging.set_verbosity(tf.logging.INFO)

# Task definition
args = {'data_dir': dm.data.sample,
        'task_dir': '/tmp/carpedm_tasks',
        'shape_store': None,
        'shape_in': (64, 64)}
task = registry.task('ocr_single_kana')(**args)

# Training Hyperparameters
num_epochs = 30
training_hparams = {'train_batch_size': 32,
                    'eval_batch_size': 1,
                    'data_format': 'channels_last',
                    'optimizer': 'sgd',
                    'learning_rate': 1e-3,
                    'momentum': 0.96,
                    'weight_decay': 2e-4,
                    'gradient_clipping': None,
                    'lr_decay_steps': None,
                    'init_dir': None,  # for pre-trained models
                    'sync': False}

# Model hyperparameters and definition
model_hparams = {}
model = registry.model('single_char_baseline')(num_classes=task.num_classes, **model_hparams)

# Unique job_id
experiment_id = 'example'
shape = re.sub(r'([,])', '_', re.sub(r'([() ])', '', str(args['shape_in'])))
job_id = os.path.join(experiment_id, shape, model.name)
task.job_id = job_id  # Used to check for first model initialization.
job_dir = os.path.join(task.task_log_dir, job_id)

# TensorFlow Configuration
sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    intra_op_parallelism_threads=0,
    gpu_options=tf.GPUOptions(force_gpu_compatible=True))
config = tf.estimator.RunConfig(session_config=sess_config,
                                model_dir=job_dir,
                                save_summary_steps=10)
hparams = tf.contrib.training.HParams(is_chief=config.is_chief,
                                      **training_hparams)

# Input and model functions
train_input_fn = task.input_fn(hparams.train_batch_size,
                               subset='train',
                               num_shards=1,
                               overwrite=False)
eval_input_fn = task.input_fn(hparams.eval_batch_size,
                              subset='dev',
                              num_shards=1,
                              overwrite=False)
model_fn = task.model_fn(model, num_gpus=0, variable_strategy='CPU',
                         num_workers=config.num_worker_replicas or 1)

# Number of training steps
train_examples = dm.data.num_examples_per_epoch(task.task_data_dir, 'train')
eval_examples = dm.data.num_examples_per_epoch(task.task_data_dir, 'dev')

if eval_examples % hparams.eval_batch_size != 0:
    raise ValueError(('validation set size (%d) must be multiple of '
                      'eval_batch_size (%d)') % (eval_examples,
                                                 hparams.eval_batch_size))

eval_steps = eval_examples // hparams.eval_batch_size
train_steps = num_epochs * ((train_examples // hparams.train_batch_size) or 1)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps)

# Estimator definition and training
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=hparams)
tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
