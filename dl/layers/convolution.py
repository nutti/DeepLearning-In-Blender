import numpy as np

from .layer_base import LayerBase


class Convolution2DLayer(LayerBase):
    def __init__(self, size, in_features, out_features, stride=1, padding=0):
        super().__init__()

        self.cache = {}
        self.params = {
            "weight": np.zeros((size, size, in_features, out_features)),
            "bias": np.zeros((1, 1, 1, out_features)),
        }
        self.grads = {
            "weight": np.zeros((size, size, in_features, out_features)),
            "bias": np.zeros((1, 1, 1, out_features)),
        }
        self.hparams = {
            "stride": stride,
            "padding": padding,
        }

    def initialize_parameters(self):
        self.params["weight"] = np.random.randn(*self.params["weight"].shape) * 0.01
        self.params["bias"] = np.zeros(self.params["bias"].shape)

    def parameters(self):
        return self.params
    
    def gradients(self):
        return self.grads

    def forward(self, x):
        weight = self.parameters["weight"]
        bias = self.parameters["bias"]
        stride = self.hparams["stride"]
        padding = self.hparams["padding"]

        assert x.shape[3] == weight.shape[2], "Shape does not match between x and weight"
        assert weight.shape[0] == weight.shape[1], "Weight shape[0] and shape[1] must be same"

        m, hi, wi, ci = x.shape
        f, f, ci, co = weight.shape

        ho = int((hi - f + 2 * padding) / stride) + 1
        wo = int((wi - f + 2 * padding) / stride) + 1

        y = np.zeros((m, ho, wo, co))
        x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)),
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
                        xm_sliced = xm[h_start:h_end, w_start:w_end]
                        weight_sliced = weight[:, :, :, c]
                        bias_sliced = bias[:, :, :, c]
                        y[i, h, w, c] = np.sum(xm_sliced * weight_sliced) + float(bias_sliced)
        
        self.cache["x"] = x

        return y

    def backward(self, dy):
        x = self.cache["x"]
        weight = self.parameters["weight"]
        bias = self.parameters["bias"]

        stride = self.hparams["stride"]
        padding = self.hparams["padding"]

        assert x.shape[3] == weight.shape[2], "Shape does not match between x and weight"
        assert weight.shape[0] == weight.shape[1], "Weight shape[0] and shape[1] must be same"

        m, hi, wi, ci = x.shape
        f, f, ci, co = weight.shape
        m, ho, wo, co = dy.shape

        dx = np.zeros((m, hi, wi, ci))
        dw = np.zeros((f, f, ci, co))
        db = np.zeros((1, 1, 1, co))

        x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                       mode="constant", constant_values=(0, 0))
        dx_pad = np.pad(dx, ((0, 0), (padding, padding), (padding, padding), (0, 0)),
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

                        xm_sliced = xm[h_start:h_end, w_start:w_end]
                        dxm[h_start:h_end, w_start:w_end] += weight[:, :, :, c] * dy[i, h, w, c]
                        dw[:, :, :, c] += xm_sliced * dy[i, h, w, c]
                        db[:, :, :, c] += dy[i, h, w, c]
            if padding != 0:
                dx[i] = dxm[padding:-padding, padding:-padding]
            else:
                dx[i] = dxm

        assert x.shape == dx.shape, "Shape does not match between x and dx" 
        
        self.grads["weight"] = dw
        self.grads["bias"] = db

        return dx
