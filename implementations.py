# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

"""
Logistic regression related functions
"""

def logistic_regression_gradient(y, tx, w, *args):
    sigmoid = 1 / (1 + np.exp(- (tx @ w)))
    return tx.T @ (sigmoid - (y + 1)/2)


def logistic_regression_loss(y, tx, w, *args):
    xtw = tx @ w    
    return np.sum(np.log(1 + np.exp(xtw)) - (1 + y) / 2 * xtw)

# Regularised logistic regression
def reg_logistic_regression_loss(y, tx, w, lambda_):
    return logistic_regression_loss(y, tx, w) + 0.5 * lambda_ * (w.T @ w)


def reg_logistic_regression_gradient(y, tx, w, lambda_):
    return logistic_regression_gradient(y, tx, w) + lambda_ * w

"""
Least squares related functions
"""
def compute_loss_least_squares(y, tx, w, *args):
    """Calculate the loss.
    """
    N = tx.shape[0]
    e = y - tx @ w
    return 1 / (2 * N) * e.T @ e

def compute_gradient_least_squares(y, tx, w, *args):
    """Compute the gradient."""
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0]

# Regularised least squares
def compute_gradient_ridge_regression(y, tx, w, lambda_):
    """Compute the gradient."""
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0] + 2 * lambda_ * w

def compute_loss_ridge(y, tx, w, lambda_):
    return np.sqrt(2 * (compute_loss_least_squares(y, tx, w) + lambda_ * w.T @ w))


"""
Function used for Stochastic gradient descent
"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, compute_gradient, lambda_=0,
                                batch_size=1):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, lambda_)
        
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size):
            sgrad = compute_gradient(y_, tx_, w, lambda_)
            w = w - gamma * sgrad

            # store w and loss
            ws.append(w)

        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l:.10f}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return w, losses[-1]

"""
Functions to optimise weights using different methods
"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_least_squares, compute_gradient_least_squares, batch_size=y.shape[0])


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_ridge, compute_gradient_ridge_regression)


def least_squares(y, tx):
    """
    Calculate the least squares solution directly.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss_least_squares(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Calculate the regularised least square solution directly
    """
    w = np.linalg.solve(tx.T @ tx + np.identity(tx.shape[1]) * lambda_ * 2 * tx.shape[0], tx.T @ y)
    return w, compute_loss_ridge(y, tx, w, lambda_)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, logistic_regression_loss,
                                       logistic_regression_gradient)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, reg_logistic_regression_loss,
                                       reg_logistic_regression_gradient, lambda_)
