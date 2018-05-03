#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#
#
# Portions of this module are copied or lightly modified from the
# Tensor2Tensor registry_test module, so here is their license:
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


"""Tests for utils.registry

References:
    Slight modification of `Tensor2Tensor registry_test`_.

..  _Tensor2Tensor registry_test: https://github.com/tensorflow/
    tensor2tensor/blob/master/tensor2tensor/utils/registry_test.py
"""

import unittest

from carpedm.util import registry
from carpedm.models.generic import Model
from carpedm.models.baseline import SingleCharBaseline


class ModelRegistryTest(unittest.TestCase):

    def setUp(self):
        registry._reset()

    def test_model_registration(self):

        @registry.register_model
        class MyModel1(Model):
            pass

        model = registry.model("my_model1")
        self.assertTrue(model is MyModel1)

    def test_named_registration(self):

        @registry.register_model("model2")
        class MyModel1(Model):
            pass

        model = registry.model("model2")
        self.assertTrue(model is MyModel1)

    def test_request_unprovided_model(self):
        with self.assertRaisesRegex(LookupError, "never registered"):
            _ = registry.model("not_provided")

    def test_duplicate_registration(self):

        @registry.register_model
        def m1():
            pass

        with self.assertRaisesRegex(LookupError, "already registered"):

            @registry.register_model("m1")
            def m2():
                pass

    def test_list_models(self):

        @registry.register_model
        def m1():
            pass

        @registry.register_model
        def m2():
            pass

        self.assertSetEqual({"m1", "m2"}, set(registry.list_models()))

    def test_snake_case(self):
        convert = registry._convert_camel_to_snake

        self.assertEqual("typical_camel_case", convert("TypicalCamelCase"))
        self.assertEqual("numbers_fuse2gether", convert("NumbersFuse2gether"))
        self.assertEqual("numbers_fuse2_gether", convert("NumbersFuse2Gether"))
        self.assertEqual("lstm_seq2_seq", convert("LSTMSeq2Seq"))
        self.assertEqual("starts_lower", convert("startsLower"))
        self.assertEqual("starts_lower_caps", convert("startsLowerCAPS"))
        self.assertEqual("caps_fuse_together", convert("CapsFUSETogether"))
        self.assertEqual("startscap", convert("Startscap"))
        self.assertEqual("s_tartscap", convert("STartscap"))


class ModelProvidedTest(unittest.TestCase):

    def setUp(self):
        from carpedm import models

    def test_access_provided_model(self):
        model = registry.model("single_char_baseline")
        self.assertTrue(model is SingleCharBaseline)


if __name__ == '__main__':
    unittest.main()
