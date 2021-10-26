# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt
from expansion import *

"""
Functions for the cross validation
"""

def build_k_indices(y, k_fold, seed):
    """
        Randomly sample k groups of indices in the dataset 
        This method was given in lab 4 of the course
        
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

    # Fix the seed and get a permutation of the row indices 
    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)

def cross_validation_visualization(lambdas, min_degree, max_degree, acc_te):
    """visualization the curves of mse_tr and mse_te.
        Parameters
            ----------
                optimization
                lambdas:
                    lambda that were used to compute the loss on the train and test set
                max_degree:
                    The losses on the train set 
                acc_te:
                    The losses on the test set
    """
    
    degrees = list(range(min_degree,max_degree+1))
    
    fig, ax = plt.subplots()
    im = ax.imshow(acc_te, cmap='viridis')

    ax.set_xticks(np.arange(len(degrees)))
    ax.set_yticks(np.arange(len(lambdas)))

    ax.set_xticklabels(degrees)
    ax.set_yticklabels(lambdas)
    plt.colorbar(im,ax=ax)

        
    plt.xlabel("Degree")
    plt.ylabel("Lambda")
    plt.title("Cross validation")
    plt.show()
    
def cross_validation_one_step(y, x, k_indices, k, max_iters, lambda_, degree, gamma,  compute_loss, compute_gradient,
                              optimization, batch_size):
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
    
    initial_w = np.zeros(x_train_cv.shape[1])

    # Compute the optimal weights and the loss according to the chosen method
    if optimization == 'sgd':
        w, loss_tr = stochastic_gradient_descent(y_train_cv, x_train_cv, initial_w, max_iters, gamma, compute_loss,
                                                 compute_gradient, lambda_=lambda_, batch_size=batch_size)
    elif optimization == 'ridge_normal_eq':
        w, loss_tr = ridge_regression(y_train_cv, x_train_cv, lambda_)
    elif optimization == 'least_squares':
        w, loss_tr = least_squares(y_train_cv, x_train_cv)
    else:
        print(f'Optimization method not supported {optimization}')
        return
    
    loss_te = compute_loss(y_test_cv, x_test_cv, w, lambda_)
    return compute_rmse(loss_tr), compute_rmse(loss_te)
    
    
def perform_cross_validation(y, tx, compute_loss, compute_gradient, max_iters, lambdas, max_degree, gamma=0.0,
                             k_fold=8, seed=1, batch_size=1, optimization='sgd'):
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
                - Optimal degree for the required method
    """

    print("Beginning cross-validation")
    
    min_degree = 2
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    range_degree = range(min_degree, max_degree+1)

    nb_lambdas = len(lambdas)
    nb_degree = len(range_degree)
    

    # Tables which will store all the loss value for all lambda-gamma combination
    rmse_tr = np.zeros((nb_lambdas, nb_degree))
    rmse_te = np.zeros((nb_lambdas, nb_degree))
    
    # iterate over all (gamma, lambda) pairs
    for ind_deg, deg in enumerate(range_degree):
        x_exp = power_exp(tx, deg)
        
        for ind_lambda, lambda_ in enumerate(lambdas):
            # List to store the loss for the current lambda and gamma pairs
            rmse_tr_tmp = []
            rmse_te_tmp = []
            
            print(f"Perform cross-validation for lambda={lambda_:.4f} and degree={deg}")
            # Perform the cross validation
            for k in range(k_fold):
                rmse_train, rmse_test  = cross_validation_one_step(y, x_exp, k_indices,
                                                                      k, max_iters, lambda_, deg, gamma,
                                                 compute_loss, compute_gradient, optimization, batch_size)
                rmse_tr_tmp.append(rmse_train)
                rmse_te_tmp.append(rmse_test)
               

            # Report the mean squared loss for the training and test sets for the current (gamma, lambda) pair 
            rmse_tr[ind_lambda, ind_deg] = np.mean(rmse_tr_tmp)
            rmse_te[ind_lambda, ind_deg] = np.mean(rmse_te_tmp)
            print(f"Actual loss for lambda={lambda_:.4f} and degree={deg}. tr={np.mean(rmse_tr_tmp)}, te={np.mean(rmse_te_tmp)}")
    
    cross_validation_visualization(lambdas, min_degree, max_degree, rmse_te)
    
    # Find the best arugments 
    argmin = rmse_te.argmin()
    best_lam_ind = argmin // max_degree
    best_degree = argmin % max_degree + min_degree

    best_lambda = lambdas[best_lam_ind]
    
    return best_lambda, best_degree