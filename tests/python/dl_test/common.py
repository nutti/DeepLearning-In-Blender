import unittest
import os
import numpy as np
import torch


LOG_DIR = "dl_test.log"


class DlTestBase(unittest.TestCase):

    name = None
    module_name = None
    log_dir = None
    file_ = None

    @classmethod
    def setUpClass(cls):
        if cls.module_name is None:
            raise ValueError("module_name must set")
        if cls.name is None:
            raise ValueError("name must set")

        cls.log_dir = "{}/{}".format(LOG_DIR, cls.module_name)
        os.makedirs(cls.log_dir, exist_ok=True)

        filename = "{}/{}.log".format(cls.log_dir, cls.name)
        cls.file_ = open(filename, "w")

    @classmethod
    def tearDownClass(cls):
        cls.file_.close()

    def setUp(self):
        self.log("========== Test: {} ==========".format(self.id()))

    def tearDown(self):
        pass

    def log(self, message):
        self.__class__.file_.write(message + "\n")
    
    def torch_to_numpy(self, torch_tensor):
        return torch_tensor.to("cpu").detach().numpy().copy()

    def numpy_to_torch(self, numpy_tensor, requires_grad=False):
        return torch.tensor(numpy_tensor.copy(), requires_grad=requires_grad)

    def assertClose(self, x, y, epsilon=1e-08):
        self.assertEqual(np.sum(np.abs(x - y) > epsilon), 0)

