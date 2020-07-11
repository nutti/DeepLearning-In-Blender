import numpy as np


def numerical_gradient(func, x):
    epsilon = 1e-4

    grad = np.zeros_like(x)

    if x.ndim == 1:
        for i in range(x.size):
            orig_xi = x[i]

            x[i] = orig_xi + epsilon
            f_plus = func(x)

            x[i] = orig_xi - epsilon
            f_minus = func(x)

            grad[i] = (f_plus - f_minus) / (2 * epsilon)
            x[i] = orig_xi
    elif x.ndim == 2:
        for i0 in range(x.shape[0]):
            for i1 in range(x.shape[1]):
                orig_xi = x[i0, i1]

                x[i0, i1] = orig_xi + epsilon
                f_plus = func(x)

                x[i0, i1] = orig_xi - epsilon
                f_minus = func(x)

                grad[i0, i1] = (f_plus - f_minus) / (2 * epsilon)
                x[i0, i1] = orig_xi

    return grad
