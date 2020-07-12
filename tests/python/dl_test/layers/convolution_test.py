import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.convolution import Convolution2DLayer

from .. import common


class Convolution2DLayerTest(common.DlTestBase):

    name = "Convolution2DLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
    
    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = Convolution2DLayer(kernel_size=2, in_channels=3, out_channels=5, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        weight = layer.parameters()["weight"].copy()
        bias = layer.parameters()["bias"].copy()
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        weight_torch = self.numpy_to_torch(weight)
        bias_torch = self.numpy_to_torch(bias)
        expect_torch = F.conv2d(x_torch, weight_torch, bias_torch, stride=1, padding=0)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)
    
    def test_foward_case_2(self):
        layer = Convolution2DLayer(kernel_size=2, in_channels=5, out_channels=3, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        weight = layer.parameters()["weight"].copy()
        bias = layer.parameters()["bias"].copy()
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        weight_torch = self.numpy_to_torch(weight)
        bias_torch = self.numpy_to_torch(bias)
        expect_torch = F.conv2d(x_torch, weight_torch, bias_torch, stride=2, padding=1)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_backward_case_1(self):
        layer = Convolution2DLayer(kernel_size=2, in_channels=3, out_channels=5, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        weight = layer.parameters()["weight"].copy()
        bias = layer.parameters()["bias"].copy()
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)
        dw_actual = layer.gradients()["weight"].copy()
        db_actual = layer.gradients()["bias"].copy()

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        weight_torch = self.numpy_to_torch(weight, requires_grad=True)
        bias_torch = self.numpy_to_torch(bias, requires_grad=True)
        y_torch = F.conv2d(x_torch, weight_torch, bias_torch, stride=1, padding=0)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dw_torch = weight_torch.grad
        db_torch = bias_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)
        dw_expect = self.torch_to_numpy(dw_torch)
        db_expect = self.torch_to_numpy(db_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertEquals(dw_expect.shape, dw_actual.shape)
        self.assertEquals(db_expect.shape, db_actual.shape)
        self.assertClose(dx_expect, dx_actual)
        self.assertClose(dw_expect, dw_actual)
        self.assertClose(db_expect, db_actual)

    def test_backward_case_2(self):
        layer = Convolution2DLayer(kernel_size=2, in_channels=5, out_channels=3, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        weight = layer.parameters()["weight"].copy()
        bias = layer.parameters()["bias"].copy()
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)
        dw_actual = layer.gradients()["weight"].copy()
        db_actual = layer.gradients()["bias"].copy()

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        weight_torch = self.numpy_to_torch(weight, requires_grad=True)
        bias_torch = self.numpy_to_torch(bias, requires_grad=True)
        y_torch = F.conv2d(x_torch, weight_torch, bias_torch, stride=2, padding=1)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dw_torch = weight_torch.grad
        db_torch = bias_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)
        dw_expect = self.torch_to_numpy(dw_torch)
        db_expect = self.torch_to_numpy(db_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertEquals(dw_expect.shape, dw_actual.shape)
        self.assertEquals(db_expect.shape, db_actual.shape)
        self.assertClose(dx_expect, dx_actual)
        self.assertClose(dw_expect, dw_actual)
        self.assertClose(db_expect, db_actual)

