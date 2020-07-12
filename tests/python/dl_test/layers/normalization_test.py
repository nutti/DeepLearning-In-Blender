import numpy as np
import torch
import torch.nn.functional as F
import random

from dl.layers.normalization import BatchNormalizationLayer

from .. import common


class BatchNormalizationLayerTest(common.DlTestBase):

    name = "BatchNormalizationLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    def tearDown(self):
        super().tearDown()

    def test_forward_case_1(self):
        layer = BatchNormalizationLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        running_mean_torch = torch.tensor([0.0, 0.0], dtype=torch.double)
        running_variance_torch = torch.tensor([0.0, 0.0], dtype=torch.double)
        expect_torch = F.batch_norm(x_torch, running_mean_torch, running_variance_torch, training=True, momentum=0, eps=1e-07)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        layer = BatchNormalizationLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, 9.0]])
        actual = layer.forward(x)

        x_torch = self.numpy_to_torch(x)
        running_mean_torch = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        running_variance_torch = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        expect_torch = F.batch_norm(x_torch, running_mean_torch, running_variance_torch, training=True, momentum=0, eps=1e-07)
        expect = self.torch_to_numpy(expect_torch)

        self.assertEquals(expect.shape, actual.shape)
        self.assertClose(expect, actual)

    def test_backward_case_1(self):
        layer = BatchNormalizationLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        running_mean_torch = torch.tensor([0.0, 0.0], dtype=torch.double)
        running_variance_torch = torch.tensor([0.0, 0.0], dtype=torch.double)
        y_torch = F.batch_norm(x_torch, running_mean_torch, running_variance_torch, training=True, momentum=0, eps=1e-07)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

    def test_backward_case_2(self):
        layer = BatchNormalizationLayer()
        layer.initialize_parameters()
        x = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, 9.0]])
        y = layer.forward(x)
        dy = np.ones(y.shape)
        dx_actual = layer.backward(dy)

        x_torch = self.numpy_to_torch(x, requires_grad=True)
        running_mean_torch = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        running_variance_torch = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        y_torch = F.batch_norm(x_torch, running_mean_torch, running_variance_torch, training=True, momentum=0, eps=1e-07)
        dy_torch = torch.ones(y_torch.shape)
        y_torch.backward(gradient=dy_torch)
        dx_torch = x_torch.grad
        dx_expect = self.torch_to_numpy(dx_torch)

        self.assertEquals(dx_expect.shape, dx_actual.shape)
        self.assertClose(dx_expect, dx_actual)

