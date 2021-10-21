# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *

"""
Functions to optimise weights using different methods
"""


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
        Function that compute the gradient descent using the least square method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            initial_w :
                Initial weight vector used to compute the gradient
            max_iters :
                Maximum number of iteration of the gradient descent
            gamma :
                Step size of the gradient descent
        Returns 
        -------
            Tuple :
                - Optimal weight vector using the least square method
                - Its corresponding loss
    """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_least_squares,
                                       compute_gradient_least_squares, batch_size=y.shape[0])


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
        Function that compute the stochastic gradient descent using the least square method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            initial_w :
                Initial weight vector used to compute the gradient
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            gamma :
                Step size of the stochastic gradient descent
        Returns 
        -------
            Tuple :
                - Optimal weight vector using the least square method
                - Its corresponding loss
    """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_least_squares,
                                       compute_gradient_least_squares)


def least_squares(y, tx):
    """
        Calculate the solution of the least square equation
        
        Parameters
        ----------
            y :
                    Outputs of the data points
            tx :
                Data points

        Returns
        -------
            Tuple :
                - Optimal weight vector which is the solution to the least square equation
                - Its corresponding loss using the least square method
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    return w, compute_loss_least_squares(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
        Calculate the regularised least square solution directly
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            lambda_ :
                Regularizer of the regularised least square method
            
        Returns
        -------
            Tuple :
                - Optimal weight vector which is the solution to the regularized least square equation
                - Its corresponding loss using the reguralized least square method
    """
    w = np.linalg.solve(tx.T @ tx + np.identity(tx.shape[1]) * lambda_ * 2 * tx.shape[0], tx.T @ y)

    return w, compute_loss_ridge(y, tx, w, lambda_)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        Function that compute the stochastic gradient descent of the logistic regression
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            initial_w :
                Initial weight vector used to compute the gradient
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            gamma :
                Step size of the stochastic gradient descent
        Returns 
        -------
            Tuple :
                - Optimal weight vector of the logistic regression
                - Its corresponding loss
    """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_logistic_regression,
                                       compute_gradient_logistic_regression)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
        Function that compute the stochastic gradient descent of the regularized logistic regression
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            initial_w :
                Initial weight vector used to compute the gradient
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            gamma :
                Step size of the stochastic gradient descent
        Returns 
        -------
            Tuple :
                - Optimal weight vector of the regularized logistic regression
                - Its corresponding loss
    """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_reg_logistic_regression,
                                       compute_gradient_reg_logistic_regression, lambda_)


"""
Logistic regression related functions
"""


def compute_gradient_logistic_regression(y, tx, w, *args):
    """
        Function that computes the gradient of weight vector w in the case of the logistic regression
        
        Parameters 
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
        Returns 
        -------
            Gradient, using the logistic regression method, computed with the parameters
            
    """
    # Sigmoid function used in the logistic regression
    sigmoid = 1 / (1 + np.exp(- (tx @ w)))
    return tx.T @ (sigmoid - 0.5 * (y + 1))


def compute_loss_logistic_regression(y, tx, w, *args):
    """
        Function that compute the loss of the weight vector w in the case of the logistic regression
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the loss
        Returns 
        -------
            Loss of the weight vector, using the logistic regression method, computed with the parameters
    """
    xtw = tx @ w
    return np.sum(np.log(1 + np.exp(xtw)) - 0.5 * (1 + y) * xtw) / tx.shape[0]


# Regularized logistic regression
def compute_loss_reg_logistic_regression(y, tx, w, lambda_):
    """
        Function that compute the loss of the weight vector w in the case of the regularized logistic regression
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the loss
            lambda_ :
                Regularizer of this regularized logistic regression
        Returns 
        -------
            Loss of the weight vector, using the regularized logistic regression method, computed with the parameters
    """
    return compute_loss_logistic_regression(y, tx, w) #+ 0.5 * lambda_ * (w.T @ w)


def compute_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """
        Function that computes the gradient of the wieght vector w in the case of the regularized logistic regression
        
        Parameters 
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
            lambda_ :
                Regularizer of this regularized logistic regression
        Returns 
        -------
            Gradient, using the regularized logistic regression method, computed with the parameters
            
    """
    return compute_gradient_logistic_regression(y, tx, w) + lambda_ * w


def predict_labels_logistic_regression(weights, x):
    """
        Function that computes the predictions for the given data of the logistic regression model with given the weights
        
        Parameters 
        ----------
            weights :
                Trained weights of the model
            y :
                Data points
        Returns 
        -------
            Predicted labels of the model for the given data
            
    """
    y = 1 / (1 + np.exp(- x @ weights))

    y[y >= 0.5] = 1
    y[y < 0.5] = -1

    return y


"""
Least squares related functions
"""


def compute_loss_least_squares(y, tx, w, *args):
    """
        Function that compute the loss of the weight vector according to the least square method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the loss
        Returns 
        -------
            Loss of the weight vector, using the least square method, computed with the parameters
    """
    # Number of data points
    N = tx.shape[0]
    # error vector
    e = y - tx @ w

    return 1 / (2 * N) * e.T @ e


def compute_gradient_least_squares(y, tx, w, *args):
    """
        Function that compute the gradient of the weight vector according to the least square method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
        Returns 
        -------
            Gradient of the weight vector, using the least square method, computed with the parameters
    """
    # error vector
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0]


def compute_gradient_ridge_regression(y, tx, w, lambda_):
    """
        Function that compute the gradient of the weight vector according to the ridge regression method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
            lambda_ :
                Regularizer of this ridge regression
        Returns 
        -------
            Gradient of the weight vector, using the ridge regression method, computed with the parameters
    """
    # error vector
    e = y - tx @ w
    return - (tx.T @ e) / y.shape[0] + 2 * lambda_ * w


def compute_loss_ridge(y, tx, w, lambda_):
    """
        Function that compute the loss of the weight vector according to the ridge regression method
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
            lambda_ :
                Regularizer of this ridge regression
        Returns 
        -------
            Loss of the weight vector, using the ridge regression method, computed with the parameters
    """
    return compute_loss_least_squares(y, tx, w) #+ lambda_ * w.T @ w


"""
Functions used for Stochastic Gradient Descent
"""


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    This function was given as part of the exercice 2.
    
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
    """
        Function that run the stochastic gradient descent and output the best weight vector and its according loss
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            initial_w :
                Initial weight vector used to compute the gradient
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            gamma :
                Step size of the stochastic gradient descent
            compute_loss :
                Function to use to compute the loss of the weight vector
            compute_gradient:
                Function to use to compute the gradient of the weight vector
            lambda_ :
                The importance of the regularizer
            batch_size :
                Size of the batch to use in the stochastic gradient descent
        Returns 
        -------
            w :
                Optimal weight vector obtained at the end of the stochastic gradient descent
            losses[-1] :
                The loss of the optimal weight vector
    """

    # Define parameters to store w and loss
    losses = []
    w = initial_w

    # start of the stochastic gradient descent
    for n_iter in range(max_iters):
        # compute the loss of the corresponding optimal weight vector
        loss = compute_loss(y, tx, w, lambda_)

        # start of the compute of the stochastic gradient for the weight vector candidate
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size):
            # stochastic gradient with particular batch of data points and its corresponding outputs
            sgrad = compute_gradient(y_, tx_, w, lambda_)

            # Gradient Descent
            w = w - gamma * sgrad

        # store loss
        losses.append(loss)
        
    return w, losses[-1]
