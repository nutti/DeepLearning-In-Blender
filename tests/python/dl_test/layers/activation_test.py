import numpy as np

from dl.layers.activation import TanhLayer, SoftmaxLayer

from .. import common

class TanhLayerTest(common.DlTestBase):

    name = "TanhLayerTest"
    module_name = __module__

    def setUp(self):
        super().setUp()
    
    def tearDown(self):
        super().tearDown()
    
    def test_foward_case_1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        expect = np.tanh(x)

        layer = TanhLayer()
        actual = layer.forward(x)

        self.assertClose(expect, actual)

    def test_forward_case_2(self):
        x = np.array([[-3.0, 4.0]])

        expect = np.tanh(x)

        layer = TanhLayer()
        actual = layer.forward(x)

        self.assertClose(expect, actual)

    def test_backward_cast_1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        dy = np.array([[0.5, -0.5], [0.5, 0.5]])

        y = np.tanh(x)
        expect = (1 - np.power(y, 2)) * dy

        layer = TanhLayer()
        y = layer.forward(x)
        actual = layer.backward(dy)

        self.assertClose(expect, actual)

    def test_backward_cast_2(self):
        x = np.array([[-3.0, 4.0]])
        dy = np.array([[1.5, -2.5]])

        y = np.tanh(x)
        expect = (1 - np.power(y, 2)) * dy

        layer = TanhLayer()
        y = layer.forward(x)
        actual = layer.backward(dy)

        self.assertClose(expect, actual)

