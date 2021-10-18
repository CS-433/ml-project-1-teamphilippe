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
    return compute_rmse(loss_tr), compute_rmse(loss_te)

def cross_validation_visualization(lambdas, mse_tr, mse_te,logy=True):
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
    
    if logy:
        plt.loglog(lambdas, mse_tr, marker=".", color='b', label='train error')
        plt.loglog(lambdas, mse_te, marker=".", color='r', label='test error')
    else:
        plt.semilogx(lambdas, mse_tr, marker=".", color='b', label='train error')
        plt.semilogx(lambdas, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("Cross validation")
    plt.legend(loc=2)
    plt.grid(True)

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

    # Tables which will store all the loss value for all lambda-gamma combination
    rmse_te = np.zeros((nb_lambdas, nb_gammas))
    rmse_tr = np.zeros((nb_lambdas, nb_gammas))

    for ind_gamma, gamma in enumerate(gammas):
        for ind_lambda, lambda_ in enumerate(lambdas):
            # List to store the loss for the current lambda and gamma pairs
            rmse_te_tmp = []
            rmse_tr_tmp = []
            
            # Perform the cross validation
            for k in range(k_fold):
                rmse_training, rmse_test  = cross_validation_one_step(y, tx, initial_w, k_indices, k, max_iters, gamma, lambda_,
                                                 compute_loss, compute_gradient, optimization, batch_size)
                rmse_te_tmp.append(rmse_test)
                rmse_tr_tmp.append(rmse_training)

            # Report the mean square loss for the training set and the test set 
            rmse_te[ind_lambda, ind_gamma] = np.mean(rmse_te_tmp)
            rmse_tr[ind_lambda, ind_gamma] = np.mean(rmse_tr_tmp)
    
    # Plot the loss of the test and the train set
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    # Find the best arugments 
    argmin = rmse_te.argmin()
    best_lam_ind = argmin // nb_gammas
    best_gam_ind = argmin % nb_gammas

    best_lambda = lambdas[best_lam_ind]
    best_gamma = gammas[best_gam_ind]
    
    return best_lambda, best_gamma

"""
Customise version of the stochastic gradient descent to show plot of the loss function 
"""
def stochastic_gradient_descent_validation(y, tx, y_te, x_te, initial_w, max_iters, gamma, compute_loss, compute_gradient, lambda_=0,
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
    ws = [initial_w]
    losses = []
    losses_test = []
    
    w = initial_w

    # start of the stochastic gradient descent
    for n_iter in range(max_iters):
        # compute the loss of the corresponding optimal weight vector
        loss = compute_loss(y, tx, w, lambda_)
        loss_te = compute_loss(y_te, x_te, w, lambda_)

        # start of the compute of the stochastic gradient for the weight vector candidate
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size):
            # stochastic gradient with particular batch of data points and its corresponding outputs
            sgrad = compute_gradient(y_, tx_, w, lambda_)

            # Gradient Descent
            w = w - gamma * sgrad

            # store w
            ws.append(w)

        # store loss
        losses.append(loss)
        losses_test.append(loss_te)
        
    #plt.plot(losses)
    #plt.plot(losses_test)
    #plt.show()
    return w, losses[-1]



def train_and_predict(y_tr, x_tr, y_te, x_te, model, seed, initial_w, max_iters, lambdas, gammas):
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
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_logistic_regression, compute_gradient_logistic_regression,
                max_iters, seed=seed, lambdas=[1], gammas=gammas)

            # Train the model on the local training set
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                compute_loss_logistic_regression, compute_gradient_logistic_regression)

        elif model == 'reg_logistic_regression':
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression,
                max_iters, seed=seed, lambdas=lambdas, gammas=gammas)

            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te,  initial_w, max_iters, best_gamma,
                                                        compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression, lambda_=best_lambda)

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
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=[1], gammas=gammas, batch_size=y_tr.shape[0])

            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                     compute_loss_least_squares,compute_gradient_least_squares,batch_size=len(x_tr))

        elif model == 'least_squares_SGD':
            # Cross validation only to find a good learning rate gamma
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=[1], gammas=gammas)

            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                      compute_loss_least_squares, compute_gradient_least_squares)

        elif model == 'least_squares':
            w, loss_mse = least_squares(y_tr, x_tr)

        elif model == 'ridge_regression':
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, seed=seed, lambdas=lambdas, gammas=gammas, optimization='ridge_normal_eq')

            w, loss_mse = ridge_regression(y_tr, x_tr, best_lambda)

        else:
            print(f'Model ({model}) not supported')
            return

        # Predict the labels on the local test set
        y_hat_te = predict_labels(w, x_te)

    return y_hat_te, w, loss_mse


def run_experiment(y, x, model, seed, ratio_split_tr, max_iters=100, lambdas=np.logspace(-10, 0, 50), gammas=[0.0001]):
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

    y_hat_te, w_opti, loss_mse = train_and_predict(y_tr, x_tr, y_te, x_te, model, seed, initial_w, max_iters, lambdas, gammas)

    # Compute the accuracy on the local test set
    accuracy_test = compute_accuracy(y_te, y_hat_te)
    f1_test = compute_f1_score(y_te, y_hat_te)
    print(f'Accuracy of {model} on the local test set : {accuracy_test:.4f}')
    print(f'F1-score of {model} on the local test set : {f1_test:.4f}')
    return accuracy_test, f1_test, w_opti
