import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.dropout import DropoutLayer

from .. import common


class DropoutLayerTest(common.DlTestBase):

    name = "DropoutLayer"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_forward_case_1(self):
        layer = DropoutLayer(drop_ratio=0.5)
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        expect = np.array([[0.0, 2.0], [0.0, 0.0]])

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = DropoutLayer(drop_ratio=0.2)
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0, 5.0], [-3.0, 4.0, 1.0]])
        actual = layer.forward(x)

        expect = np.array([[1.0, 2.0, 0.0], [-3.0, 0.0, 0.0]])

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_case_1(self):
        layer = DropoutLayer(drop_ratio=0.5)
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones_like(y)
        dx_actual = layer.backward(dy)

        dx_expect = np.array([[0.0, 1.0], [0.0, 0.0]])

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_case_2(self):
        layer = DropoutLayer(drop_ratio=0.2)
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0, 5.0], [-3.0, 4.0, 1.0]])
        y = layer.forward(x)
        dy = np.ones_like(y)
        dx_actual = layer.backward(dy)

        dx_expect = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

