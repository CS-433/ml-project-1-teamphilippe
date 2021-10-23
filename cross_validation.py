# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt

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

    # Fix the seed and get a perfumation of the row indices 
    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)

def cross_validation_visualization(lambdas, rmse_tr, rmse_te, logy=False):
    """visualization the curves of mse_tr and mse_te.
        Parameters
            ----------
                optimization
                lambdas:
                    lambda that were used to compute the loss on the train and test set
                mse_tr:
                    The losses on the train set 
                mse_te:
                    The losses on the test set
                logy:
                    True if the plot should be a log-log plot, log-lin plot otherwise
    """
    # This method comes from the helper "plots.py" and was given in lab 4
    
    # Plots either a log log plot or a semilog plot depending on what is asked
    if logy:
        plt.loglog(lambdas, rmse_tr, marker=".", color='b', label='train error')
        plt.loglog(lambdas, rmse_te, marker=".", color='r', label='test error')
    else:
        plt.semilogx(lambdas, rmse_tr, marker=".", color='b', label='train error')
        plt.semilogx(lambdas, rmse_te, marker=".", color='r', label='test error')
        
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("Cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    
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
    x_test_cv = x[k_indices[k]]
    y_test_cv = y[k_indices[k]]

    x_train_cv = np.delete(x, k_indices[k], axis=0)
    y_train_cv = np.delete(y, k_indices[k], axis=0)
    

    # Compute the optimal weights and the loss according to the chosen method
    if optimization == 'sgd':
        w, loss_tr = stochastic_gradient_descent(y_train_cv, x_train_cv, initial_w, max_iters, gamma, compute_loss,
                                                 compute_gradient, lambda_=lambda_, batch_size=batch_size)
    elif optimization == 'ridge_normal_eq':
        w, loss_tr = ridge_regression(y_train_cv, x_train_cv, lambda_)

    else:
        print(f'Optimization method not supported {optimization}')
        return
    
    #loss_te = compute_loss(y_test_cv, x_test_cv, w, lambda_)
    acc_test = compute_accuracy(y_test_cv, predict_labels(w, x_test_cv))
    acc_train = compute_accuracy(y_train_cv, predict_labels(w, x_train_cv))
    #return compute_rmse(loss_tr), compute_rmse(loss_te)
    return acc_train, acc_test
    
def perform_cross_validation(y, tx, compute_loss, compute_gradient, max_iters, initial_w , 
                             k_fold=8, seed=1, lambdas=np.logspace(-4, 0, 30), gammas=[0.05], batch_size=1, optimization='sgd'):
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

    print("Beginning cross-validation")
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    nb_lambdas = len(lambdas)
    nb_gammas = len(gammas)

    # Tables which will store all the loss value for all lambda-gamma combination
    rmse_te = np.zeros((nb_lambdas, nb_gammas))
    rmse_tr = np.zeros((nb_lambdas, nb_gammas))

    # iterate over all (gamma, lambda) pairs
    for ind_gamma, gamma in enumerate(gammas):
        for ind_lambda, lambda_ in enumerate(lambdas):
            # List to store the loss for the current lambda and gamma pairs
            rmse_te_tmp = []
            rmse_tr_tmp = []
            
            print(f"Perform cross-validation for lambda={lambda_:.4f} and gamma={gamma:.4f}")
            # Perform the cross validation
            for k in range(k_fold):
                rmse_training, rmse_test  = cross_validation_one_step(y, tx, initial_w, k_indices,
                                                                      k, max_iters, gamma, lambda_,
                                                 compute_loss, compute_gradient, optimization, batch_size)
                
                rmse_te_tmp.append(rmse_test)
                rmse_tr_tmp.append(rmse_training)

            # Report the mean squared loss for the training and test sets for the current (gamma, lambda) pair 
            rmse_te[ind_lambda, ind_gamma] = np.mean(rmse_te_tmp)
            rmse_tr[ind_lambda, ind_gamma] = np.mean(rmse_tr_tmp)
    
    # Plot the loss of the test and the train set
    if(nb_gammas==1):
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    # Find the best arugments 
    argmin = rmse_te.argmax()
    best_lam_ind = argmin // nb_gammas
    best_gam_ind = argmin % nb_gammas

    best_lambda = lambdas[best_lam_ind]
    best_gamma = gammas[best_gam_ind]
    
    return best_lambda, best_gamma