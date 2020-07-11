import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.activation import(
    ReluLayer,
    SigmoidLayer,
    TanhLayer,
    SoftmaxLayer,
)

from .. import common


class ReluLayerTest(common.DlTestBase):

    name = "ReluLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
    
    def tearDown(self):
        super().tearDown()
    
    def test_foward_case_1(self):
        layer = ReluLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.relu(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = ReluLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = F.relu(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = ReluLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.relu(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_cast_2(self):
        layer = ReluLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = F.relu(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)


class SigmoidLayerTest(common.DlTestBase):

    name = "SigmoidLayer"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = SigmoidLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.sigmoid(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = SigmoidLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.sigmoid(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = SigmoidLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.sigmoid(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_cast_2(self):
        layer = SigmoidLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.sigmoid(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)


class TanhLayerTest(common.DlTestBase):

    name = "TanhLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.tanh(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.tanh(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.tanh(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_cast_2(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.tanh(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)


class SoftmaxLayerTest(common.DlTestBase):

    name = "SoftmaxLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_foward_case_1(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.tanh(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        expect_torch = torch.tanh(x_torch)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.tanh(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_cast_2(self):
        layer = TanhLayer()
        layer.initialize_parameters()
        x = np.array([[-3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        y_torch = torch.tanh(x_torch)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_actual.shape, dx_expect.shape)
        self.assertClose(dx_expect, dx_actual)

