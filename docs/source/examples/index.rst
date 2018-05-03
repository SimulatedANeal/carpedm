Examples
========

Single Character Task
---------------------

Below is an example Task definition for a single character recognition task
and the corresponding import in ``__init__.py`` for accessing the task through
the registry.

For more details on Task definition and default properties, please refer to
the :ref:`Tasks` documentation.

ocr.py
~~~~~~
.. literalinclude:: ../../../carpedm/tasks/ocr.py
    :language: python

tasks.__init__.py
~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../../carpedm/tasks/__init__.py
    :language: python

Baseline Model
--------------

baseline.py
~~~~~~~~~~~
.. literalinclude:: ../../../carpedm/models/baseline.py
    :language: python

models.__init__.py
~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../../carpedm/models/__init__.py
    :language: python


.. _main:

Using Tasks and Models
----------------------

Below is a minimal ``main.py`` example for getting started training a model using the Task interface.
For an in-depth description, please refer to the guide :ref:`training`.

.. literalinclude:: main.py
    :language: python

