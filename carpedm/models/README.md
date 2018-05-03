# Constructing Models

This directory contains built-in and user-defined models.

## Adding a new model

1. Create a new file (see, e.g., baseline.py).
2. Write your model class inheriting from one of the base `Model` classes:
    * `TFModel`: Use a TensorFlow backend.
3. Decorate it with `registry.register_model`.
4. Import it in `__init__.py`.

## Accessing models
To access a registered (built-in or added using the steps above) model, in a ``main.py`` file, for example, use:
```python
import carpedm as dm  # Important: registers models.
from carpedm.util import registry

...  # define hyperparameters, etc.

model = registry.model('registered_model_name')(**hparams)
```

