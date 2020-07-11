import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.element_wise import(
    AddLayer,
    MulLayer,
)

from .. import common


class AddLayerTest(common.DlTestBase):

    name = "AddLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = AddLayer()
        layer.initialize_parameters()
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x2 = np.array([[-3.0, 4.0], [1.0, -0.5]])
        actual = layer.forward(x1, x2)

        x1_torch = self.numpy_to_torch(x1)
        x2_torch = self.numpy_to_torch(x2)
        expect_torch = x1_torch + x2_torch
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = AddLayer()
        layer.initialize_parameters()
        x1 = np.array([[-3.0, 4.0]])
        x2 = np.array([[0.0, 1.0]])
        actual = layer.forward(x1, x2)

        x1_torch = self.numpy_to_torch(x1)
        x2_torch = self.numpy_to_torch(x2)
        expect_torch = x1_torch + x2_torch
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = AddLayer()
        layer.initialize_parameters()
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x2 = np.array([[-3.0, 4.0], [1.0, -0.5]])
        y = layer.forward(x1, x2)
        dy = np.ones(y.shape)
        dx1_actual, dx2_actual = layer.backward(dy)

        x1_torch = self.numpy_to_torch(x1, requires_grad=True)
        x2_torch = self.numpy_to_torch(x2, requires_grad=True)
        y_torch = x1_torch + x2_torch
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx1_torch = x1_torch.grad
        dx2_torch = x2_torch.grad
        dx1_expect = self.torch_to_numpy(dx1_torch)
        dx2_expect = self.torch_to_numpy(dx2_torch)

        self.assertEquals(dx1_actual.shape, dx1_expect.shape)
        self.assertEquals(dx2_actual.shape, dx2_expect.shape)
        self.assertClose(dx1_expect, dx1_actual)
        self.assertClose(dx2_expect, dx2_actual)

    def test_backward_cast_2(self):
        layer = AddLayer()
        layer.initialize_parameters()
        x1 = np.array([[-3.0, 4.0]])
        x2 = np.array([[0.0, 1.0]])
        y = layer.forward(x1, x2)
        dy = np.ones(y.shape)
        dx1_actual, dx2_actual = layer.backward(dy)

        x1_torch = self.numpy_to_torch(x1, requires_grad=True)
        x2_torch = self.numpy_to_torch(x2, requires_grad=True)
        y_torch = x1_torch + x2_torch
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx1_torch = x1_torch.grad
        dx2_torch = x2_torch.grad
        dx1_expect = self.torch_to_numpy(dx1_torch)
        dx2_expect = self.torch_to_numpy(dx2_torch)

        self.assertEquals(dx1_actual.shape, dx1_expect.shape)
        self.assertEquals(dx2_actual.shape, dx2_expect.shape)
        self.assertClose(dx1_expect, dx1_actual)
        self.assertClose(dx2_expect, dx2_actual)


class MulLayerTest(common.DlTestBase):

    name = "MulLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = MulLayer()
        layer.initialize_parameters()
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x2 = np.array([[-3.0, 4.0], [1.0, -0.5]])
        actual = layer.forward(x1, x2)

        x1_torch = self.numpy_to_torch(x1)
        x2_torch = self.numpy_to_torch(x2)
        expect_torch = x1_torch * x2_torch
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = MulLayer()
        layer.initialize_parameters()
        x1 = np.array([[-3.0, 4.0]])
        x2 = np.array([[0.0, 1.0]])
        actual = layer.forward(x1, x2)

        x1_torch = self.numpy_to_torch(x1)
        x2_torch = self.numpy_to_torch(x2)
        expect_torch = x1_torch * x2_torch
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = MulLayer()
        layer.initialize_parameters()
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x2 = np.array([[-3.0, 4.0], [1.0, -0.5]])
        y = layer.forward(x1, x2)
        dy = np.ones(y.shape)
        dx1_actual, dx2_actual = layer.backward(dy)

        x1_torch = self.numpy_to_torch(x1, requires_grad=True)
        x2_torch = self.numpy_to_torch(x2, requires_grad=True)
        y_torch = x1_torch * x2_torch
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx1_torch = x1_torch.grad
        dx2_torch = x2_torch.grad
        dx1_expect = self.torch_to_numpy(dx1_torch)
        dx2_expect = self.torch_to_numpy(dx2_torch)

        self.assertEquals(dx1_actual.shape, dx1_expect.shape)
        self.assertEquals(dx2_actual.shape, dx2_expect.shape)
        self.assertClose(dx1_expect, dx1_actual)
        self.assertClose(dx2_expect, dx2_actual)

    def test_backward_cast_2(self):
        layer = MulLayer()
        layer.initialize_parameters()
        x1 = np.array([[-3.0, 4.0]])
        x2 = np.array([[0.0, 1.0]])
        y = layer.forward(x1, x2)
        dy = np.ones(y.shape)
        dx1_actual, dx2_actual = layer.backward(dy)

        x1_torch = self.numpy_to_torch(x1, requires_grad=True)
        x2_torch = self.numpy_to_torch(x2, requires_grad=True)
        y_torch = x1_torch * x2_torch
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx1_torch = x1_torch.grad
        dx2_torch = x2_torch.grad
        dx1_expect = self.torch_to_numpy(dx1_torch)
        dx2_expect = self.torch_to_numpy(dx2_torch)

        self.assertEquals(dx1_actual.shape, dx1_expect.shape)
        self.assertEquals(dx2_actual.shape, dx2_expect.shape)
        self.assertClose(dx1_expect, dx1_actual)
        self.assertClose(dx2_expect, dx2_actual)
