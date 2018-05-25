*******************************
CarpeDM: Sieze the Data Manager
*******************************

.. image:: https://pypip.in/version/carpedm/badge.png
    :target: https://pypi.python.org/pypi/carpedm/
    :alt: Latest Version

.. image:: https://travis-ci.org/SimulatedANeal/carpedm.png
    :target: https://travis-ci.org/SimulatedANeal/carpedm
    :alt: Continuous Integration Testing

.. image:: https://pypip.in/license/carpedm/badge.png
    :target: https://pypi.python.org/pypi/carpedm/
    :alt: License

.. image:: https://readthedocs.org/projects/carpedm/badge/?version=latest
    :target: http://carpedm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

`Docs <http://carpedm.readthedocs.io/en/latest/>`_
| `Install Guide <http://carpedm.readthedocs.io/en/latest/install.html>`_
| `Tutorial <http://carpedm.readthedocs.io/en/latest/guides/usage.html>`_

Description
===========
*CarpeDM* is a general library for downloading, viewing, and manipulating image data.
Originally developed as a ChARacter shaPE Data Manager, CarpeDM aims to make Japanese character shape (字形) data
and other image datasets more accessible to machine learning researchers.

.. csv-table:: Datasets Currently Available for Download
    :header: "ID", "Dataset"
    :widths: 10, 80

    "pmjtc", "| `Pre-Modern Japanese Text Character Shapes Dataset (日本古典籍字形データセット) <http://codh.rois.ac.jp/char-shape/>`_,
    | provided by the Center for Open Data in the Humanities (CODH)."

Though still in the early stages of development, a high-level interface is also provided for

* Automatic model-ready data generation.
* Flexible training of models with a variety of deep learning frameworks.

Currently supported deep learning frameworks:

* `TensorFlow <https://www.tensorflow.org/>`_
