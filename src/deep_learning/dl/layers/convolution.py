import numpy as np
from collections import OrderedDict

from .layer_base import LayerBase


class Convolution2DLayer(LayerBase):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, padding=0):
        super().__init__()

        self.cache = {}

        self.params = OrderedDict()
        self.params["weight"] = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.params["bias"] = np.zeros((out_channels, ))

        self.grads = OrderedDict()
        self.grads["weight"] = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.grads["bias"] = np.zeros((out_channels, ))

        self.hparams = {
            "stride": stride,
            "padding": padding,
        }

    def id(self):
        return "Convolution2D"

    def initialize_parameters(self, weight=None, bias=None):
        if weight is None:
            self.params["weight"] = np.random.randn(*self.params["weight"].shape) * 0.01
        else:
            assert self.params["weight"].shape == weight.shape, "shape of 'weight' must be {}, but {}".format(self.params["weight"].shape, weight.shape)
            self.params["weight"] = weight

        if bias is None:
            self.params["bias"] = np.zeros(self.params["bias"].shape)
        else:
            assert self.params["bias"].shape == bias.shape, "shape of 'bias' must be {}, but {}".format(self.params["bias"].shape, bias.shape)
            self.params["bias"] = bias

    def parameters(self):
        return self.params
    
    def gradients(self):
        return self.grads

    def forward(self, x):
        weight = self.params["weight"]
        bias = self.params["bias"]
        stride = self.hparams["stride"]
        padding = self.hparams["padding"]

        assert x.shape[1] == weight.shape[1], "Shape does not match between x.shape[1] ({}) and weight.shape[1] ({})".format(x.shape[1], weight.shape[1])
        assert weight.shape[2] == weight.shape[3], "Weight shape[2] ({}) and shape[3] ({}) must be same".format(weight.shape[2], weight.shape[3])

        m, ci, hi, wi = x.shape
        co, ci, f, f = weight.shape

        ho = int((hi - f + 2 * padding) / stride) + 1
        wo = int((wi - f + 2 * padding) / stride) + 1

        y = np.zeros((m, co, ho, wo))
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode="constant", constant_values=(0, 0))
        
        for i in range(m):
            xm = x_pad[i]
            for h in range(ho):
                h_start = h * stride
                h_end = h * stride + f
                for w in range(wo):
                    w_start = w * stride
                    w_end = w * stride + f
                    for c in range(co):
                        xm_sliced = xm[:, h_start:h_end, w_start:w_end]
                        weight_sliced = weight[c, :, :, :]
                        bias_sliced = bias[c]
                        y[i, c, h, w] = np.sum(xm_sliced * weight_sliced) + float(bias_sliced)
        
        self.cache["x"] = x

        return y

    def backward(self, dy):
        x = self.cache["x"]
        weight = self.params["weight"]
        bias = self.params["bias"]

        stride = self.hparams["stride"]
        padding = self.hparams["padding"]

        assert x.shape[1] == weight.shape[1], "Shape does not match between x.shape[1] ({}) and weight.shape[1] ({})".format(x.shape[1], weight.shape[1])
        assert weight.shape[2] == weight.shape[3], "Weight shape[2] ({}) and shape[3] ({}) must be same".format(weight.shape[2], weight.shape[3])

        m, ci, hi, wi = x.shape
        co, ci, f, f = weight.shape
        m, co, ho, wo = dy.shape

        dx = np.zeros((m, ci, hi, wi))
        dw = np.zeros((co, ci, f, f))
        db = np.zeros((co,))

        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode="constant", constant_values=(0, 0))
        dx_pad = np.pad(dx, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                        mode="constant", constant_values=(0, 0))

        for i in range(m):
            xm = x_pad[i]
            dxm = dx_pad[i]
            for h in range(ho):
                for w in range(wo):
                    for c in range(co):
                        h_start = h * stride
                        h_end = h * stride + f
                        w_start = w * stride
                        w_end = w * stride + f

                        xm_sliced = xm[:, h_start:h_end, w_start:w_end]
                        dxm[:, h_start:h_end, w_start:w_end] += weight[c, :, :, :] * dy[i, c, h, w]
                        dw[c, :, :, :] += xm_sliced * dy[i, c, h, w]
                        db[c] += dy[i, c, h, w]
            if padding != 0:
                dx[i] = dxm[:, padding:-padding, padding:-padding]
            else:
                dx[i] = dxm

        assert x.shape == dx.shape, "Shape does not match between x and dx" 
        assert weight.shape == dw.shape, "Shape does not match between weight and dw"
        assert bias.shape == db.shape, "Shape does not match between bias and db"
        
        self.grads["weight"] = dw
        self.grads["bias"] = db

        return dx
