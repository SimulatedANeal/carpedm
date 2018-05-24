#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Base task class.

Todo:
    * Get rid of ``model_fn`` dependency on ``input_fn``.
    * LONG TERM: Training methods other than TensorFlow Estimator.
"""
import abc
import os
import re

import tensorflow as tf
from tensorflow.contrib.training import GreedyLoadBalancingStrategy

from carpedm.data.lang import CharacterSet, JapaneseUnicodes
from carpedm.data.meta import MetaLoader
from carpedm.data.ops import shard_batch
from carpedm.nn.util import TOWER_NAME
from carpedm.util.train import config_optimizer
from carpedm.util.train import local_device_setter, make_hooks
from carpedm.util.train import compute_global_grads_loss, group_train_op


# Special tokens
GO_TOKEN = "<GO>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Task(object):
    """Abstract class for Tasks."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, task_dir, test_split='hnsd00000',
                 dev_split=0.1, dev_factor=1, dataset_format='tfrecords',
                 num_shards=8, num_threads=8, shape_store=None, shape_in=None,
                 vocab_size=None, min_frequency=0, seed=None, **kwargs):
        """Initializer.

        Args:
            data_dir (str): Directory where raw data is stored.
            task_dir (str): Top-level directory for storing tasks data
                and results.
            test_split (float or str): Either the ratio of all data
                to use for testing or specific bibliography ID(s). Use
                comma-separated IDs for multiple books.
            dev_split (float or str): Either the ratio of training data
                to use for dev/val or specific bibliography ID(s). Use
                comma-separated IDs for multiple books.
            dev_factor: (int): Size of development set should be
                divisible by this value. Useful for training on
                multiple GPUs.
            dataset_format (str): Base storage unit for the dataset.
            vocab_size (int): Maximum vocab size.
            min_frequency (int): Minimum frequency of type to be
                included in vocab.
            shape_store (tuple or None): Size to which images are resized
                for storage, if needed, e.g. for TFRecords. The default
                is to not perform any resize. Please see this `note on
                image shape`_ for more information.
            shape_in (tuple or None): Size to which images are resized
                by interpolation or padding before being input to a
                model. Please see this `note on image shape`_ for
                more information.
            num_shards (int): Number of sharded output files.
            num_threads (int): Number of threads to run in parallel.
            seed (int or None): Number for seeding rng.
            **kwargs: Unused arguments.

        """
        self._task_dir = task_dir
        self._test_split = test_split
        self._dev_split = dev_split
        self._dataset_format = dataset_format
        self._num_shards = num_shards
        self._num_threads = num_threads
        self._shape_store = shape_store
        self._shape_in = shape_in
        self.job_id = "_"
        self._meta = MetaLoader(
            data_dir=data_dir,
            test_split=self._test_split, dev_split=self._dev_split,
            dev_factor=dev_factor, vocab_size=vocab_size,
            min_freq=min_frequency, reserved=self.reserved,
            charset=self.character_set, image_scope=self.image_scope,
            seq_len=self.sequence_length, seq_maxlen=self.max_sequence_length,
            seed=seed)

    # ====================== BEGIN TASK INTERFACE ==================== #

    @abc.abstractmethod
    def results(self, loss, tower_features, tower_preds, tower_targets,
                is_training):
        """Accumulates predictions, computes metrics, and determines
        the tensors to log and/or visualize.

        Args:
            loss (tf.float): Global loss.
            tower_features (list of dict): Tower feature dicts.
            tower_preds (list): Tower predictions.
            tower_targets (list of tf.Tensor): Tower targets.
            is_training (bool): The model is training.

        Returns:
            dict: The tensors to log
            dict: All predictions
            dict: Evaluation metrics

        """

    @abc.abstractmethod
    def loss_fn(self, features, model_output, targets, is_training):
        """Computes an appropriate loss for the tasks.

        Must be implemented in subclass.

        Args:
            features (dict): Additional features for computing loss.
            model_output (tf.Tensor or dict of tf.Tensor): Model output
                used for computing the batch loss, e.g. class logits.
            targets (tf.Tensor): Ground truth targets.
            is_training (bool): The model is training.

        Returns:
            tf.Tensor: Losses of type 'int32' and shape [batch_size, 1]

        """

    @abc.abstractmethod
    def regularization(self, hparams):
        """

        Args:
            hparams: Hyperparameters, e.g. weight_decay

        Returns:

        """

    @property
    def sequence_length(self):
        """If max_sequence_length is None, this gives the deterministic
        length of a sequence, else the minimum sequence length.

        Only used if ``image_scope == 'seq'``.

        Returns:
            int or None:

        """
        return None

    @property
    def max_sequence_length(self):
        """Maximum sequence length.

        Only used if ``image_scope == 'seq'``.

        Returns:
            int or None:

        """
        return None

    @property
    def character_set(self):
        """The Japanese characters (e.g. kana, kanji) of interest.

        Preset character sets may include the following component sets:

            * hiragana
            * katakana
            * kana
            * kanji
            * punct (punctuation)
            * misc

        Returns:
            CharacterSet: The character set.

        """
        return JapaneseUnicodes(charset='all')

    @property
    def reserved(self):
        """Reserved tokens for the tasks.

        The index of each token in the returned tuple will be used as
        its integer ID.

        Returns:
            tuple: The reserved characters

        """
        return PAD_TOKEN, UNK_TOKEN

    @property
    def num_classes(self):
        """Total number of output nodes, includes reserved tokens."""
        return self._meta.vocab.get_num_classes()

    @property
    @abc.abstractmethod
    def target(self):
        """Determines the value against which predictions are compared.

        For a list of possible targets, refer to
        carpedm.data.util.ImageMeta.generate_features()

        Returns:
            str: feature key for the target

        """

    @property
    @abc.abstractmethod
    def image_scope(self):
        """Portion of original image for each example.

        Available scopes are 'char', 'seq', 'line', 'page'.

        Returns:
            str: Task image scope

        """

    @property
    @abc.abstractmethod
    def chunk(self):
        """When creating a dataset, instead of using the original image,
        extract non-overlapping chunks of size `image_shape` and the
        corresponding features from the original image on a regular
        grid. The original image is padded to divide evenly by
        `image_shape`.

        Note: currently only objects that are entirely contained in
        the block will have its features propagated.

        Returns:
            bool:

        """

    @property
    @abc.abstractmethod
    def character(self):
        """When creating a dataset, tell the meta_loader to generate
        character features, e.g. label, bbox.

        Returns:
            bool: Use character features.

        """

    @property
    @abc.abstractmethod
    def line(self):
        """When creating a dataset, tell the meta_loader to generate
        line features, e.g. bbox.

        Returns:
            bool: Use line features.

        """

    @property
    @abc.abstractmethod
    def label(self):
        """When creating a dataset, generate character labels.

        Returns:
            bool: Use character labels

        """

    @property
    @abc.abstractmethod
    def bbox(self):
        """When creating a dataset, generate appropriate bounding boxes
        for the tasks (determined by e.g. self.character, self.line).

        Returns:
            bool: Use bounding boxes.

        """

    @property
    @abc.abstractmethod
    def sparse_labels(self):
        """Generate labels as a SparseTensor, e.g. for CTC loss.

        Returns:
            (bool): Use sparse labels.

        """
        return False

    # ====================== END TASK INTERFACE ====================== #

    @property
    def task_data_dir(self):
        """Directory where tasks data is stored.

        Returns:
            str

        """
        shape = re.sub(
            r'([,])', '_', re.sub(r'([() ])', '', str(self._shape_store)))
        data_split = "test={}_dev={}".format(self._test_split, self._dev_split)
        dir_path = os.path.join(self._task_dir, self.task_id, 'data',
                                self._dataset_format, shape, data_split)
        return dir_path

    @property
    def task_log_dir(self):
        return os.path.join(self._task_dir, self.task_id, 'results')

    @property
    def task_id(self):
        num_classes = self._meta.vocab.get_num_classes() - len(self.reserved)
        return "{}_{}-{}".format(self.__class__.__name__,
                                 self.character_set.name,
                                 num_classes)

    def get_class_labels(self, as_unicode=False):
        return self._meta.vocab.types(as_unicode)

    def input_fn(self, batch_size, subset, num_shards, overwrite=False):
        """Returns (sharded) batches of data.

        Args:
            batch_size (int): The batch_size
            subset (str): The subset to use. One of {train, dev, test}.
            num_shards (int): Number of data_shards to produce.
            overwrite (bool): Overwrite existing data.

        Returns:
            (list): Features of length num_shards.
            (list): Labels of length num_shards.

        """

        dataset = self._meta.generate_dataset(
            out_dir=self.task_data_dir, subset=subset, target_id=self.target,
            num_shards=self._num_shards, num_threads=self._num_threads,
            format_store=self._dataset_format, shape_store=self._shape_store,
            shape_in=self._shape_in, sparse_labels=self.sparse_labels,
            chunk=self.chunk, character=self.character, line=self.line,
            label=self.label, bbox=self.bbox, overwrite=overwrite)

        self._original_format = dataset.format

        def _input_fn():

            feature_batch, label_batch = dataset.make_batch(batch_size)

            if num_shards <= 1:
                # No GPU available or only 1 GPU.
                return [feature_batch], [label_batch]
            else:
                return shard_batch(feature_batch, label_batch,
                                   batch_size, num_shards)

        return _input_fn

    def model_fn(self, model, variable_strategy, num_gpus, num_workers,
                 devices=None):
        """ Model function used by TensorFlow Estimator class.

        Args:
            model (pmjtc.models.generic.Model): The models to run.
            variable_strategy (str): Where to locate variable
                operations, either 'CPU' or 'GPU'.
            num_gpus (int): Number of GPUs to use, if available.
            devices (tuple): Specific devices to use. If provided,
                overrides num_gpus.
            num_workers (int): Parameter for distributed training.

        Returns:

        """

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        if not devices:
            devices = range(num_devices)

        def _model_fn(features, labels, mode, params):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            tower_features = features
            tower_targets = labels
            tower_losses = []
            tower_gradvars = []
            tower_preds = []

            data_format = params.data_format
            if not data_format:
                if num_gpus == 0:
                    data_format = 'channels_last'
                else:
                    data_format = 'channels_first'

            if data_format != self._original_format:
                if self._original_format == 'channels_last':
                    # Computation requires channels_first.
                    axes_order = [0, 3, 1, 2]
                else:
                    # Computation requires channels_last.
                    axes_order = [0, 2, 3, 1]
            else:
                axes_order = None

            for i in devices:
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=GreedyLoadBalancingStrategy(
                            num_gpus, tf.contrib.training.byte_size_load_fn))
                else:
                    raise ValueError("variable_strategy must be CPU or GPU.")
                with tf.variable_scope(model.name, reuse=bool(i != 0)):
                    with tf.name_scope(TOWER_NAME + '_%d' % i) as name_scope:
                        with tf.device(device_setter):
                            loss, gradvars, preds = _tower_fn(
                                features=tower_features[i],
                                targets=tower_targets[i],
                                data_format=data_format,
                                axes_order=axes_order,
                                is_training=is_training,
                                params=params)
                            tower_losses.append(loss)
                            tower_gradvars.append(gradvars)
                            tower_preds.append(preds)
                            if i == 0:
                                update_ops = tf.get_collection(
                                    tf.GraphKeys.UPDATE_OPS, name_scope)

            # Device that runs the ops to apply global gradient updates.
            if variable_strategy == 'GPU':
                consolidation_device = '/gpu:0'
            else:
                consolidation_device = '/cpu:0'
            with tf.device(consolidation_device):
                gradvars, loss = compute_global_grads_loss(tower_gradvars,
                                                           tower_losses)
                optimizer = config_optimizer(params)
                train_op = group_train_op(optimizer, gradvars, update_ops)
                tensors_to_log, predictions, metrics = self.results(
                    loss, tower_features, tower_preds, tower_targets,
                    is_training)
                train_hooks = make_hooks(tensors_to_log, optimizer,
                                         num_workers, params)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)

        def _tower_fn(features, targets, data_format, axes_order, is_training,
                      params):
            preds = model.forward_pass(
                features, data_format, axes_order, is_training)

            if params.init_dir:
                if os.path.exists(params.init_dir):
                    variable_mapping = model.initialize_pretrained(
                        params.init_dir)
                    # First initialization only
                    if not os.path.exists(
                            os.path.join(self.task_log_dir, self.job_id)):
                        tf.train.init_from_checkpoint(params.init_dir,
                                                      variable_mapping)
                else:
                    print("Initialization directory %s does not exist."
                          % params.init_dir, "Using default initialization.")

            loss = tf.reduce_mean(
                self.loss_fn(features, preds, targets, is_training))
            loss += self.regularization(params)

            # gradient
            model_params = tf.trainable_variables()
            gradient = tf.gradients(loss, model_params)
            if params.gradient_clipping:
                cc = params.gradient_clipping
                gradient = [tf.clip_by_value(grad, -cc, cc)
                            for grad in gradient]
            gradvars = zip(gradient, model_params)
            return loss, gradvars, preds

        return _model_fn
