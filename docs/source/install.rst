.. _install-guide:

Installation
============

Recommended Environments
------------------------

The following versions of Python can be used: 3.4, 3.5, 3.6.

We recommend setting up a virtual environment with Python 3.6 for using or developing CarpeDM.

We use `virtualenv <https://virtualenv.pypa.io/en/stable/installation/>`_,
but you could use `Conda <https://conda.io/docs/user-guide/install/index.html>`_, etc.

.. code-block:: bash

    $ virtualenv -p /path/to/python3 ~/.virtualenvs/carpedm
    $ source ~/.virtualenvs/carpedm/bin/activate

Or, for Conda:

.. code-block:: bash

    $ conda create --name carpedm python=3.6
    $ conda activate carpedm

.. note::
    CarpeDM is built and tested on MacOS. We cannot guarantee that it works on other environments, including Windows and Linux.

Dependencies
------------

Before installing CarpeDM, we recommend to upgrade ``setuptools`` if you are using an old one::

  $ pip install -U setuptools

The following Python packages are required to install CarpeDM.
The latest version of each package will automatically be installed if missing.

* `TensorFlow <https://www.tensorflow.org/>`__ 1.5+
* `Numpy <https://www.numpy.org>`__ 1.14+
* `Pillow <http://python-pillow.org/>`__ 5.1+

The following packages are optional dependencies.

* Plot and images support

    * `matplotlib <https://matplotlib.org>`_ 2.1.2, 2.2.2

Install CarpeDM
---------------

Install CarpeDM via pip
~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing the latest release of CarpeDM with pip::

    $ pip install carpedm

.. note::

    Any optional dependencies can be added after installing CarpeDM.
    Please refer to :ref:`optional`.

Install CarpeDM from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install a development version of CarpeDM from a cloned Git repository::

  $ git clone https://github.com/SimulatedANeal/carpedm.git
  $ cd carpedm
  $ python setup.py develop

.. _optional:

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Support Plotting and Viewing Images
```````````````````````````````````

Using the following (see :ref:`meta-loader`)

.. code-block:: python

    MetaLoader.view_images()
    MetaLoader.data_stats(which_stats=('frequency'))

require ``matplotlib``. We recommend installing it with pip::

    $ pip install matplotlib

Uninstall CarpeDM
-----------------

Use pip to uninstall CarpeDM::

    $ pip uninstall carpedm

Upgrade CarpeDM
---------------
Just use ``pip`` with ``-U`` option::

  $ pip install -U carpedm

FAQ
---
