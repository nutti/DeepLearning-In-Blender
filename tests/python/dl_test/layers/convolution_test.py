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
        x_torch = torch.Tensor(np.random.rand(2, 3, 4, 4))
        weight_torch = torch.Tensor(np.random.rand(5, 3, 2, 2))
        bias_torch = torch.Tensor(np.zeros((5,)))
        expect_torch = F.conv2d(x_torch, weight_torch, bias_torch,
                                stride=1, padding=0)

        x = self.torch_to_numpy(x_torch.permute(0, 2, 3, 1))
        weight = self.torch_to_numpy(weight_torch.permute(2, 3, 1, 0))
        bias = self.torch_to_numpy(bias_torch.reshape(1, 1, 1, -1))
        expect = self.torch_to_numpy(expect_torch.permute(0, 2, 3, 1))

        layer = Convolution2DLayer(stride=1, padding=0)
        actual = layer.forward(x, weight, bias)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    
    def test_foward_case_2(self):
        x_torch = torch.Tensor(np.random.rand(1, 5, 3, 3))
        weight_torch = torch.Tensor(np.random.rand(3, 5, 2, 2))
        bias_torch = torch.Tensor(np.zeros((3,)))
        expect_torch = F.conv2d(x_torch, weight_torch, bias_torch,
                                stride=2, padding=1)

        x = self.torch_to_numpy(x_torch.permute(0, 2, 3, 1))
        weight = self.torch_to_numpy(weight_torch.permute(2, 3, 1, 0))
        bias = self.torch_to_numpy(bias_torch.reshape(1, 1, 1, -1))
        expect = self.torch_to_numpy(expect_torch.permute(0, 2, 3, 1))

        layer = Convolution2DLayer(stride=2, padding=1)
        actual = layer.forward(x, weight, bias)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_backward_case_1(self):
        x_torch = torch.Tensor(np.random.rand(2, 3, 4, 4))
        weight_torch = torch.Tensor(np.random.rand(5, 3, 2, 2))
        bias_torch = torch.Tensor(np.zeros((5,)))
        dy_torch = torch.Tensor(np.random.rand(2, 5, 3, 3))
        dx_expect_torch = F.grad.conv2d_input(x_torch.shape, weight_torch,
                                              dy_torch, stride=1, padding=0)
        dw_expect_torch = F.grad.conv2d_weight(x_torch, weight_torch.shape,
                                               dy_torch, stride=1, padding=0)

        x = self.torch_to_numpy(x_torch.permute(0, 2, 3, 1))
        weight = self.torch_to_numpy(weight_torch.permute(2, 3, 1, 0))
        bias = self.torch_to_numpy(bias_torch.reshape(1, 1, 1, -1))
        dy = self.torch_to_numpy(dy_torch.permute(0, 2, 3, 1))
        dx_expect = self.torch_to_numpy(dx_expect_torch.permute(0, 2, 3, 1))
        dw_expect = self.torch_to_numpy(dw_expect_torch.permute(2, 3, 1, 0))

        layer = Convolution2DLayer(stride=1, padding=0)
        layer.forward(x, weight, bias)
        dx_actual, dw_actual, db_actual = layer.backward(dy)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertEquals(dw_expect.shape, dw_actual.shape)
        self.assertClose(dx_expect, dx_actual, epsilon=1e-6)
        self.assertClose(dw_expect, dw_actual, epsilon=1e-6)


    def test_backward_case_2(self):
        x_torch = torch.Tensor(np.random.rand(1, 5, 3, 3))
        weight_torch = torch.Tensor(np.random.rand(3, 5, 2, 2))
        bias_torch = torch.Tensor(np.zeros((3,)))
        dy_torch = torch.Tensor(np.random.rand(1, 3, 2, 2))
        dx_expect_torch = F.grad.conv2d_input(x_torch.shape, weight_torch,
                                              dy_torch, stride=2, padding=1)
        dw_expect_torch = F.grad.conv2d_weight(x_torch, weight_torch.shape,
                                               dy_torch, stride=2, padding=1)

        x = self.torch_to_numpy(x_torch.permute(0, 2, 3, 1))
        weight = self.torch_to_numpy(weight_torch.permute(2, 3, 1, 0))
        bias = self.torch_to_numpy(bias_torch.reshape(1, 1, 1, -1))
        dy = self.torch_to_numpy(dy_torch.permute(0, 2, 3, 1))
        dx_expect = self.torch_to_numpy(dx_expect_torch.permute(0, 2, 3, 1))
        dw_expect = self.torch_to_numpy(dw_expect_torch.permute(2, 3, 1, 0))

        layer = Convolution2DLayer(stride=2, padding=1)
        layer.forward(x, weight, bias)
        dx_actual, dw_actual, db_actual = layer.backward(dy)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertEquals(dw_expect.shape, dw_actual.shape)
        self.assertClose(dx_expect, dx_actual, epsilon=1e-6)
        self.assertClose(dw_expect, dw_actual, epsilon=1e-6)

