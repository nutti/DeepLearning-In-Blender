import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.pooling import AveragePooling2DLayer, MaxPooling2DLayer

from .. import common


class MaxPooling2DLayerTest(common.DlTestBase):

    name = "MaxPooling2DLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = MaxPooling2DLayer(kernel_size=2, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.max_pool2d(x_torch, kernel_size=2, stride=1, padding=0)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_foward_case_2(self):
        layer = MaxPooling2DLayer(kernel_size=2, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.max_pool2d(x_torch, kernel_size=2, stride=2, padding=1)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_backward_case_1(self):
        layer = MaxPooling2DLayer(kernel_size=2, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.max_pool2d(x_torch, kernel_size=2, stride=1, padding=0)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_case_2(self):
        layer = MaxPooling2DLayer(kernel_size=2, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.max_pool2d(x_torch, kernel_size=2, stride=2, padding=1)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)


class AveragePooling2DLayerTest(common.DlTestBase):

    name = "AveragePooling2DLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = AveragePooling2DLayer(kernel_size=2, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.avg_pool2d(x_torch, kernel_size=2, stride=1, padding=0)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_foward_case_2(self):
        layer = AveragePooling2DLayer(kernel_size=2, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.avg_pool2d(x_torch, kernel_size=2, stride=2, padding=1)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual, epsilon=1e-6)

    def test_backward_case_1(self):
        layer = AveragePooling2DLayer(kernel_size=2, stride=1, padding=0)
        layer.initialize_parameters()
        x = np.random.rand(2, 3, 4, 4)
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.avg_pool2d(x_torch, kernel_size=2, stride=1, padding=0)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_case_2(self):
        layer = AveragePooling2DLayer(kernel_size=2, stride=2, padding=1)
        layer.initialize_parameters()
        x = np.random.rand(1, 5, 3, 3)
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.avg_pool2d(x_torch, kernel_size=2, stride=2, padding=1)
        dy_torch  = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

