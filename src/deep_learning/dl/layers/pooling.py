import numpy as np

from .layer_base import LayerBase


class MaxPooling2DLayer(LayerBase):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()

        self.cache = {}

        self.hparams = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }

    def id(self):
        return "MaxPooling2D"

    def forward(self, x):
        stride = self.hparams["stride"]
        padding = self.hparams["padding"]
        f = self.hparams["kernel_size"]

        m, c, hi, wi = x.shape

        ho = int((hi - f + 2 * padding) / stride) + 1
        wo = int((wi - f + 2 * padding) / stride) + 1

        y = np.zeros((m, c, ho, wo))
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
                    xm_sliced = xm[:, h_start:h_end, w_start:w_end]
                    xm_sliced_max = np.max(xm_sliced, axis=(1, 2))
                    y[i, :, h, w] = xm_sliced_max
        
        self.cache["x"] = x

        return y

    def backward(self, dy):
        x = self.cache["x"]
        stride = self.hparams["stride"]
        padding = self.hparams["padding"]
        f = self.hparams["kernel_size"]

        m, ci, hi, wi = x.shape
        m, co, ho, wo = dy.shape

        assert ci == co, "Channel does not match between mask and dy" 

        dx = np.zeros((m, ci, hi, wi))

        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode="constant", constant_values=(0, 0))
        dx_pad = np.pad(dx, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                        mode="constant", constant_values=(0, 0))

        for i in range(m):
            xm_pad = x_pad[i]
            dxm = dx_pad[i]
            for h in range(ho):
                for w in range(wo):
                    for c in range(co):
                        h_start = h * stride
                        h_end = h * stride + f
                        w_start = w * stride
                        w_end = w * stride + f
                        xm_sliced = xm_pad[c, h_start:h_end, w_start:w_end]
                        xm_sliced_max = np.max(xm_sliced)
                        mask = xm_sliced == xm_sliced_max
                        dxm[c, h_start:h_end, w_start:w_end] += mask * dy[i, c, h, w]
            if padding != 0:
                dx[i] = dxm[:, padding:-padding, padding:-padding]
            else:
                dx[i] = dxm

        return dx

