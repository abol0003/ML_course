# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    poly = np.ones((N, degree + 1))
    for d in range(1, degree + 1):
        poly[:, d] = x ** d
    return poly
