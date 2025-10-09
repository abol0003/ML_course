# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    #raise NotImplementedError
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.inv(a).dot(b)
    mse = np.mean((y - tx.dot(w)) ** 2)
    return w, mse
