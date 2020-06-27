import numpy as np

from dl.layers.linear import LinearLayer

from .. import common

class LinearLayerTest(common.DlTestBase):

    name = "LinearLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
    
    def tearDown(self):
        super().tearDown()
    
    def test_foward_case_1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight = np.array([[2.0, 3.0], [-1.0, -4.0]])
        bias = np.array([[0.5, -0.5]])

        expect = np.dot(weight, x) + bias

        layer = LinearLayer()
        actual = layer.forward(x, weight, bias)

        self.assertClose(expect, actual)
    
    def test_forward_case_2(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight = np.array([[3.0, 4.0]])
        bias = np.array([[1.0]])

        expect = np.dot(weight, x) + bias 

        layer = LinearLayer()
        actual = layer.forward(x, weight, bias)

        self.assertClose(expect, actual)

    def test_backward_case_1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight = np.array([[2.0, 3.0], [-1.0, -4.0]])
        bias = np.array([[0.5, -0.5]])
        dy = np.array([[0.5, -0.5], [0.5, 0.5]])

        dx_expect = np.dot(weight.T, dy)
        dw_expect = np.dot(dy, x.T)
        db_expect = np.sum(dy, axis=1, keepdims=True)

        layer = LinearLayer()
        layer.forward(x, weight, bias)    # Cache
        dx_actual, dw_actual, db_actual = layer.backward(dy)

        self.assertClose(dx_expect, dx_actual)
        self.assertClose(dw_expect, dw_actual)
        self.assertClose(db_expect, db_actual)

    def test_backward_case_2(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight = np.array([[3.0, 4.0]])
        bias = np.array([[1.0]])
        dy = np.array([[0.5, -0.5]])

        dx_expect = np.dot(weight.T, dy)
        dw_expect = np.dot(dy, x.T)
        db_expect = np.sum(dy, axis=1, keepdims=True)

        layer = LinearLayer()
        layer.forward(x, weight, bias)    # Cache
        dx_actual, dw_actual, db_actual = layer.backward(dy)

        self.assertClose(dx_expect, dx_actual)
        self.assertClose(dw_expect, dw_actual)
        self.assertClose(db_expect, db_actual)

