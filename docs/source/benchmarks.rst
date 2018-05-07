.. _benchmarks:

Benchmarks
==========

Single Kana OCR
---------------

Running the example :ref:`main.py <main>` for the full PMJTC dataset (171,944 training examples, 131 character classes, as of 2 May 2018)

* On a 2017 MacBook Pro:
    * Generating the (train & dev) data: 1 hour, 20 minutes
    * Training the model for 5 epochs: 2 hours, 27 minutes
    * Dev Accuracy: 94.67%

* On a Linux Machine using 1 Titan X (Pascal) GPU:
    * Generating the (train & dev) data: 31 minutes
    * Training the model for 5 epochs: 21 minutes
    * Dev Accuracy: 95.23%
