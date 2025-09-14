import numpy as np


def compute_subgradient_mae(y, tx, w,gamma):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient and loss
    # ***************************************************
    e=y-tx.dot(w)
    loss=np.mean(np.abs(e))
    subgrad=-(tx.T.dot(np.sign(e)))/len(y) #sign(e) to deal with the hint
    # raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: update w by subgradient
    # ***************************************************
    #raise NotImplementedError
    w=w-gamma*subgrad
    return w, loss