Basic Usage
-----------

Getting Started
~~~~~~~~~~~~~~~
There is some sample data provided, accessed as follows::

    from carpedm.data import sample as PATH_TO_SAMPLE_DATA

This small dataset is useful for getting started and debugging purposes.

Full datasets can be downloaded with::

    $ download_data -d <download/to/this/directory> -i <dataset-id>

It may take a while. For a list of available dataset IDs, use::

    $ download_data -h

Exploring the Data
~~~~~~~~~~~~~~~~~~
To quickly load and review data for a task, use the :py:class:`carpedm.data.meta.MetaLoader` class directly. Here are some example datasets that vary each image's scope and the characters included.

.. literalinclude:: ../examples/data_exploration.py
    :language: python
    :lines: 17-23

Note that these objects only store the metadata for images in the dataset, so they are relatively time and space efficient.
Assuming ``matplotlib`` is installed (see :ref:`optional`), you can use ``view_images`` to actually load and view images within the dataset.
Or use ``generate_dataset`` to save training data for a machine learning algorithm.
For example:

.. literalinclude:: ../examples/data_exploration.py
    :language: python
    :lines: 26-31

.. note::

    Currently, ``view_images`` does not work in a Jupyter notebook instance.

.. _training:

Training a Model
~~~~~~~~~~~~~~~~

The MetaLoader class on its own is useful for rapid data exploration, but
the :ref:`Tasks` module provides a high-level interface for the entire training pipeline,
from loading the raw data and automatically generating model-ready datasets, to actually training and evaluating a model.

Next, we will walk through a simple example that uses the provided single
character recognition task and a simple baseline Convolutional Neural Network model.

First, let's set our TensorFlow verbosity so we can see the training progress.

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 19,24-25

Next, we'll initialize our single kana recognition task

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 21,26-32

Most of the Task functionality, such as the target ``character_set``,
``sequence_length`` (if we're looking at character sequences ``image_scope == 'seq'``), or ``loss_fn``
is encapsulated in the class definition. However, there are some **REQUIRED** run-time task arguments:
``data_dir`` and ``task_dir`` tell the task where to find the raw data, and where to store task-specific data/results, respectively.
The other **optional** run-time arguments ``shape_store`` and ``shape_in`` determine the size of images when they are stored on disk
and fed into our neural network, respectively. If ``shape_store`` or ``shape_in`` are not provided, the original image size is used.

.. caution::

    Using the default for ``shape_in`` may break a model expecting fixed-size input.

For more information and a full list of optional arguments, please refer to the :ref:`Tasks` API.

A task can be accessed from the registry with the appropriate task ID.
By default, the ID for a stored task is a "snake_cased" version of the task class name.
Custom tasks can be added to the registry using the ``@registry.register_model`` decorator, importing the new class in ``tasks.__init__``,
and importing ``carpedm``, more specifically, the ``carpedm.tasks`` package.

Now let's define our hyper-parameters for training and our model.

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 22,33-50

The ``training_hparams`` above represent the minimal set that *must* be defined for training to run. In practice, you
may want to use a tool like `argparse <https://docs.python.org/3/library/argparse.html>`_ and define some defaults
so you don't have to explicitly define each one manually every time.
Accessing and registering models is similar to the process for tasks (see
`here <https://github.com/SimulatedANeal/carpedm/blob/master/carpedm/models/README.md>`_ for more details).

The ``baseline_cnn`` model is fully defined except for the number of classes to predict, so it doesn't take any hyper-parameters.

To distinguish this model from others, we should define a unique ``job_id``,
which can then be used in some boilerplate TensorFlow configuration.

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 52-69

We include ``shape_in`` in the job ID to avoid conflicts with loading models meant for images of different sizes.
Although we don't do so here for simplicity, it would also be a good idea to include training hyperparameter settings in the job ID,
as those are not represented in ``model.name``.

Now comes the important part: defining the input and model functions used by a TensorFlow Estimator.

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 71-81

As we can see, the Task interface makes this extremely easy!
The appropriate data subset for the task is generated (and saved) *once* automatically when ``task.input_fn`` is called.
You can overwrite previously saved data by setting the ``overwrite`` parameter to True.
The ``num_shards`` parameter can be used for training in parallel, e.g. on multiple GPUs.

``model_fn`` is a bit more complicated under the hood, but its components are simple:

* It uses ``model.forward_pass`` to generate predictions,
* ``task.loss_fn`` to train the model
* and ``task.results`` for compiling results.

I don't assume access to any GPUs, hence the values for ``num_gpus`` and ``variable_strategy``.
``variable_strategy`` tells the training manager where to collect and update variables. You can ignore the ``num_workers``
parameter, unless you want to use special distributed training, e.g. on `Google Cloud <https://cloud.google.com/products/machine-learning/>`_.

.. note::
    The ``input_fn`` definitions must come before the ``model_fn`` definition because ``model_fn`` relies on a variable, ``original_format``, defined in ``input_fn``.
    This dependence will likely be removed in future versions.

We're almost ready to train. We just need to tell it how long to train,

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 83-93

define our training manager,

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 99

and hit the train button!

.. literalinclude:: ../examples/main.py
    :language: python
    :lines: 100

Putting it all together, we have a very minimal :ref:`main.py <main>` module for training models.
Running it took **8 minutes** on a MacBook Pro, which includes data generation and training the model.
At the end of 30 epochs, it achieved a development set accuracy of **65.27%**. Not great, but this example only uses the small sample dataset (1,447 training examples).
And considering the **70** character classes and **4.19%** majority class for this task and specific dataset, we are already doing much better than chance!

Running this same code for the *full* currently available PMJTC dataset takes much longer but---as you would expect when
adding more data---achieves a higher accuracy (see :ref:`benchmarks`). Though certainly indicative of the benefit of more data,
note that the accuracies presented in the benchmarks are not a fair comparison to the one above for two reasons:

    1. There are more kana character classes in the full dataset: **131**
    2. The development sets on which accuracies are reported are different.

Conclusion
~~~~~~~~~~

I hope that this guide has introduced the basics of using CarpeDM and encourages you to define your own models and tasks,
and conduct enriching research on Pre-modern Japanese Text Characters and beyond!

Seize the Data Manager!
