#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
#
# Portions of this module are taken or lightly modified from the
# Tensor2Tensor registry module, so here is their license:
#
# Copyright 2018 The Tensor2Tensor Authors.
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


"""Registry for models and tasks.

Define a new models by subclassing models.Model and register it:

.. code-block:: python

    @registry.register_model
    class MyModel(models.Model):
        ...

Access by snake-cased name: ``registry.model("my_model")``.

See all the models registered: ``registry.list_models()``.

References:
    1. Lightly modified `Tensor2Tensor registry`_.

..  _Tensor2Tensor registry: https://github.com/tensorflow/
    tensor2tensor/blob/master/tensor2tensor/util/registry.py

"""
import re


_MODELS = {}
_TASKS = {}


# Camel case to snake case util
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r"\1_\2", name)
    return _all_cap_re.sub(r"\1_\2", s1).lower()


def _reset():
    for ctr in [_MODELS, _TASKS]:
        ctr.clear()


def default_name(obj_class):
    """Convert class name to the registry's default name for the class.

    Args:
        obj_class: the name of a class

    Returns:
        The registry's default name for the class.
    """
    return _convert_camel_to_snake(obj_class.__name__)


def default_object_name(obj):
    """Convert object to the registry's default name for the object class.

    Args:
        obj: an object instance

    Returns:
        The registry's default name for the class of the object.
    """
    return default_name(obj.__class__)


def register_model(name=None):
    """ Register a models. ``name`` defaults to class name snake-cased."""

    def decorator(model_cls, registration_name=None):
        """Registers & returns model_cls."""
        model_name = registration_name or default_name(model_cls)
        if model_name in _MODELS:
            raise LookupError("Model %s already registered." % model_name)
        model_cls.REGISTERED_NAME = model_name
        _MODELS[model_name] = model_cls
        return model_cls

    # Handle if decorator was used without parens
    if callable(name):
        model_cls = name
        return decorator(model_cls, registration_name=default_name(model_cls))

    return lambda model_cls: decorator(model_cls, name)


def model(name):
    """Retrieve a model by name."""
    if name not in _MODELS:
        raise LookupError("Model %s never registered. Available models:\n %s"
                          % (name, "\n".join(list_models())))

    return _MODELS[name]


def list_models():
    return list(sorted(_MODELS))


def register_task(name=None):
    """Register a Task. ``name`` defaults to cls name snake-cased."""

    def decorator(t_cls, registration_name=None):
        """Registers & returns t_cls with registration_name or default."""
        t_name = registration_name or default_name(t_cls)
        if t_name in _TASKS:
            raise LookupError("Task %s already registered." % t_name)

        _TASKS[t_name] = t_cls
        t_cls.name = t_name
        return t_cls

    # Handle if decorator was used without parens
    if callable(name):
        t_cls = name
        return decorator(t_cls, registration_name=default_name(t_cls))

    return lambda t_cls: decorator(t_cls, name)


def task(name):
    """Retrieve a task by name."""

    if name not in _TASKS:
        all_task_names = sorted(list_tasks())
        error_lines = ["%s not in the set of supported tasks:" % name
                       ] + all_task_names
        error_msg = "\n  * ".join(error_lines)
        raise LookupError(error_msg)
    return _TASKS[name]


def list_tasks():
    return list(_TASKS)


def display_list_by_prefix(names_list, starting_spaces=0):
    """Creates a help string for ``names_list`` grouped by prefix."""
    cur_prefix, result_lines = None, []
    space = " " * starting_spaces
    for name in sorted(names_list):
        split = name.split("_", 1)
        prefix = split[0]
        if cur_prefix != prefix:
            result_lines.append(space + prefix + ":")
            cur_prefix = prefix
        result_lines.append(space + "  * " + name)
    return "\n".join(result_lines)


def help_string():
    """Generate help string with contents of registry."""
    help_str = """
    Registry contents:
    ------------------
    
      Models:
    %s
    
      Tasks:
    %s
    """
    m, tasks = [
        display_list_by_prefix(entries, starting_spaces=4)
        for entries in [
            list_models(),
            list_tasks()
        ]
    ]
    return help_str % (m, tasks)
