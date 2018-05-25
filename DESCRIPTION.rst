*******************************
CarpeDM: Sieze the Data Manager
*******************************

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
