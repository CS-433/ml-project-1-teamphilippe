# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


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
            
def compute_gradient_least_squares(y, tx, w, *args):
    """Compute the gradient."""
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0]

def compute_gradient_ridge_regression(y, tx, w, lambda_):
    """Compute the gradient."""
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0]+2*lambda_*w

def compute_gradient_logistic_regression(y,tx,w,lambda_):
    raise NotImplementedError
    
def compute_loss_least_squares(y, tx, w):
    raise NotImplementedError
    
def compute_loss_ridge_regression(y, tx, w):
    raise NotImplementedError
    
def compute_loss_logistic_regression(y, tx, w):
    raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function, compute_gradient, lambda_=0):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size):
            sgrad = compute_gradient(y_, tx_, w, lambda_)

            w = w - gamma * sgrad

            # store w and loss
            ws.append(w)
            
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l:.10f}, w0={w0:.4f}, w1={w1:.4f}".format(
          bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
    
def least_squares(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
    
def ridge_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
