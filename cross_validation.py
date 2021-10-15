# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *

"""
Functions for the cross validation
"""


def build_k_indices(y, k_fold, seed):
    """
        Randomly sample k groups of indices in the dataset 
        
        Parameters
        ----------
            y :
                Outputs of the data points
            k_fold :
                Number of groups to use
            seed :
                Seed to initialize the RNG
        Returns 
        -------
            Array of k_fold arrays containing the indices of the k groups of samples
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)

    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation_one_step(y, x, initial_w, k_indices, k, max_iters, gamma, lambda_, compute_loss, compute_gradient,
                              optimization='sgd', batch_size=1):
    """
        Perform one step of the cross validation :
        train the model on the local training dataset and evaluate it
        on the leaved out fold.

        Parameters
        ----------
            y :
                Outputs of the data points
            x :
                Data points
            initial_w :
                Initial weights
            k_indices :
                Groups of indices of the folds
            k :
                Number of folds
            max_iters :
                Maximum number of iterations in the SGD
            gamma :
                Learning rate
            lambda_ :
                Coefficient of the regularizer
            compute_loss :
                Function to compute the model loss
            compute_gradient :
                Function to compute the model gradient
            optimization :
                Type of method to optimize the weights (either 'sgd' or 'ridge_normal_eq')
            batch_size :
                Batch size to use in the SGD
        Returns
        -------
            RMSE loss on the leaved out fold
    """
    # Build the local train/test folds
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]

    x_train = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0)

    if optimization == 'sgd':
        w, loss_tr = stochastic_gradient_descent(y_train, x_train, initial_w, max_iters, gamma, compute_loss,
                                                 compute_gradient, lambda_=lambda_, batch_size=batch_size)
    elif optimization == 'ridge_normal_eq':
        w, loss_tr = ridge_regression(y_train, x_train, lambda_)

    else:
        print(f'Optimization method not supported {optimization}')
        return

    loss_te = compute_loss(y_test, x_test, w, lambda_)
    return compute_rmse(loss_te)


def perform_cross_validation(y, tx, compute_loss, compute_gradient, max_iters, k_fold=4, seed=1,
                             lambdas=np.logspace(-4, 0, 30), gammas=[0.05], batch_size=1, optimization='sgd'):
    """
        Perform cross validation to find the best lambda (regulariser) and gamma (learning rate).
        
        Parameters
        ----------
            optimization
            y :
                Outputs of the data points
            tx :
                Data points
            compute_loss:
                Function to use to compute the loss
            compute_gradient: 
                Function to use to compute the gradient
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            k_fold: 
                Number of passes for the cross-validation
            seed:
                Seed for random splitting
            lambdas: 
                Range of values to find the best lambda
            gammas :
                Step size of the stochastic gradient descent
            batch_size:
                Number of samples in each batch of the SGD
            optimization :
                Type of method to optimize the weights (either 'sgd' or 'ridge_normal_eq')
        Returns 
        -------
            Tuple :
                - Optimal lambda for the required method
                - Optimal gamma for the required method
                - RMSE according for the different values of the hyperparameters
    """
    # Default parameters
    initial_w = np.zeros(tx.shape[1])

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    nb_lambdas = len(lambdas)
    nb_gammas = len(gammas)

    # Table which will store all the loss value for all lambda-gamma combination
    rmse_te = np.zeros((nb_gammas, nb_lambdas))

    for ind_gamma, gamma in enumerate(gammas):
        for ind_lambda, lambda_ in enumerate(lambdas):
            rmse_te_tmp = []

            # Perform the cross validation
            for k in range(k_fold):
                rmse = cross_validation_one_step(y, tx, initial_w, k_indices, k, max_iters, gamma, lambda_,
                                                 compute_loss, compute_gradient, optimization, batch_size)
                rmse_te_tmp.append(rmse)

            # Report the mean square loss
            rmse_te[ind_gamma, ind_lambda] = np.mean(rmse_te_tmp)

    # Find the best arugments 
    argmin = rmse_te.argmin()
    best_gam_ind = argmin // nb_lambdas
    best_lam_ind = argmin % nb_lambdas

    best_lambda = lambdas[best_lam_ind]
    best_gamma = gammas[best_gam_ind]

    return best_lambda, best_gamma, rmse_te


def train_and_predict(y_tr, x_tr, x_te, model, seed, initial_w, max_iters, lambdas, gammas):
    """
        Train the given model and predict the labels of the local test set.
        If necessary for the model, performs a cross validation on the hyperparameters.

        Parameters
        ----------
            y_tr :
                Outputs of the training data points
            x_tr :
                Training data points
            x_te :
                Test data points
            model :
                String, model to use
            seed :
                Seed to initialize the RNG
            initial_w :
                Initial weights
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            lambdas :
                The different coefficients to try in the cross validation
            gammas :
                The different learning rates to try in the cross validation

        Returns
        -------
        Tuple :
                - Predicted labels on the given local test set
                - Optimal weights found
    """
    if model in ['logistic_regression', 'reg_logistic_regression']:

        if model == 'logistic_regression':
            # Cross validate the regularizer coefficient and the learning rate
            best_lambda, best_gamma, rmse_te_cv = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_logistic_regression, compute_gradient_logistic_regression,
                max_iters, seed=seed, lambdas=[1], gammas=gammas)

            # Train the model on the local training set
            w, loss_mse = logistic_regression(y_tr, x_tr, initial_w, max_iters, best_gamma)

        elif model == 'reg_logistic_regression':
            best_lambda, best_gamma, rmse_te_cv = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression,
                max_iters, seed=seed, lambdas=lambdas, gammas=gammas)

            w, loss_mse = reg_logistic_regression(y_tr, x_tr, best_lambda, initial_w, max_iters, best_gamma)

        else:
            print(f'Model ({model}) not supported')
            return

        # Predict the labels on the local test set
        y_hat_te = predict_labels_logistic_regression(w, x_te)

    else:
        if model == 'least_squares_GD':
            # Full gradient descent is equivalent to SGD with
            # batch size = N
            # Cross validation only to find a good learning rate gamma
            best_lambda, best_gamma, rmse_te_cv = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=[1], gammas=gammas, batch_size=y_tr.shape[0])

            w, loss_mse = least_squares_GD(y_tr, x_tr, initial_w, max_iters, best_gamma)

        elif model == 'least_squares_SGD':
            # Cross validation only to find a good learning rate gamma
            best_lambda, best_gamma, rmse_te_cv = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=[1], gammas=gammas)

            w, loss_mse = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, best_gamma)

        elif model == 'least_squares':
            w, loss_mse = least_squares(y_tr, x_tr)

        elif model == 'ridge_regression':
            best_lambda, best_gamma, rmse_te_cv = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=lambdas, gammas=gammas, optimization='ridge_normal_eq')

            w, loss_mse = ridge_regression(y_tr, x_tr, best_lambda)

        else:
            print(f'Model ({model}) not supported')
            return

        # Predict the labels on the local test set
        y_hat_te = predict_labels(w, x_te)

    return y_hat_te, w


def run_experiment(y, x, model, seed, ratio_split_tr, max_iters=100, lambdas=np.logspace(-4, 0, 30), gammas=[0.05]):
    """
        Perform a complete pre-processing, cross-validation, training, testing experiment.

        Parameters
        ----------
            y :
                Outputs of the data points
            x :
                Data points
            model :
                Model to train (string)
            seed :
                Seed to initialize the RNG
            ratio_split_tr :
                Ratio of samples to keep in the training set
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            lambdas :
                The different coefficients to try in the cross validation
            gammas :
                The different learning rates to try in the cross validation
        Returns
        -------
        Tuple :
            - The accuracy of the model on the local test set
            - Optimal weights found
    """
    # Split the training set into a local training set and a local test set
    x_tr, y_tr, x_te, y_te = split_data(x, y, ratio=ratio_split_tr, seed=seed)

    # Standardize all the features
    x_tr, mean_x_tr, std_x_tr = standardize(x_tr)
    x_te, mean_x_te, std_x_te = standardize(x_te)

    # Initialize some settings
    initial_w = np.zeros((x_tr.shape[1]))

    y_hat_te, w_opti = train_and_predict(y_tr, x_tr, x_te, model, seed, initial_w, max_iters, lambdas, gammas)

    # Compute the accuracy on the local test set
    accuracy_test = compute_accuracy(y_te, y_hat_te)

    print(f'Accuracy of {model} on the local test set : {accuracy_test:.4f}')

    return accuracy_test, w_opti
