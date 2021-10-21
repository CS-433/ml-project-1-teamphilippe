import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt
from cross_validation import *

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
    losses = []
    losses_test = []
    
    w = initial_w

    # start of the stochastic gradient descent
    for n_iter in range(max_iters):
        # compute the loss of the corresponding optimal weight vector on the training and test sets
        loss = compute_loss(y, tx, w, lambda_)
        loss_te = compute_loss(y_te, x_te, w, lambda_)

        # start of the compute of the stochastic gradient for the weight vector candidate
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size):
            # stochastic gradient with particular batch of data points and its corresponding outputs
            sgrad = compute_gradient(y_, tx_, w, lambda_)

            # Gradient Descent
            w = w - gamma * sgrad

        # store losses
        losses.append(loss)
        losses_test.append(loss_te)
        
    # Plot the test and training losses to see the convergence
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
            if(len(gammas)>1):
                best_lambda, best_gamma = perform_cross_validation(
                    y_tr, x_tr,
                    compute_loss_logistic_regression, compute_gradient_logistic_regression,
                    max_iters, initial_w, seed=seed, lambdas=[0.0], gammas=gammas)
            else:
                # If we have only one gamma, do not need to perform cross-validation
                best_lambda, best_gamma = 0.0, gammas[0]

            # Train the model with the best parameters on teh local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                compute_loss_logistic_regression, compute_gradient_logistic_regression)

        elif model == 'reg_logistic_regression':
            # Perform cross validation to find the best regulariser and learning rate
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression,
                max_iters, initial_w, seed=seed, lambdas=lambdas, gammas=gammas)
            
            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                            compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression, lambda_=best_lambda)

        # Predict the labels on the local test set
        y_hat_te = predict_labels_logistic_regression(w, x_te)
    else:
        if model == 'least_squares_GD':
            # Full gradient descent is equivalent to SGD with
            # batch size = N, i.e. the size of the training set
            
            # Cross validation only to find a good learning rate gamma (lambda is not used in least_squares GD)
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, initial_w, seed=seed, lambdas=[0.0], gammas=gammas, batch_size=y_tr.shape[0])

            # Train the model with the best parameters on the local training set + plot the loss curves. We train with all the samples 
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                     compute_loss_least_squares,compute_gradient_least_squares, batch_size=len(x_tr))

        elif model == 'least_squares_SGD':
            # Cross validation only to find a good learning rate gamma (lambda is not used in least_squares GD)
            best_lambda, best_gamma = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, initial_w, seed=seed, lambdas=[0.0], gammas=gammas)

            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, initial_w, max_iters, best_gamma,
                                                      compute_loss_least_squares, compute_gradient_least_squares)

        elif model == 'least_squares':
            # Find the least squares solution via the normal equation
            w, loss_mse = least_squares(y_tr, x_tr)

        elif model == 'ridge_regression':
            # Cross validation to find the best lambda (no gradient descent => gamma is not used)
            best_lambda, _ = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_ridge, compute_gradient_ridge_regression,
                max_iters, initial_w, seed=seed, lambdas=lambdas, gammas=[0.0], optimization='ridge_normal_eq')

            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = ridge_regression(y_tr, x_tr, best_lambda)

        else:
            print(f'Model ({model}) not supported')
            return

        # Predict the labels on the local test set
        y_hat_te = predict_labels(w, x_te)

    return y_hat_te, w, loss_mse


def run_experiment(y, x, model, seed, ratio_split_tr, max_iters=1000, lambdas=np.logspace(-4, 0, 50), gammas=[0.00095]):
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

    exp_x_tr = build_expansion(x_tr)
    exp_x_te = build_expansion(x_te)
    
    # Standardize all the features
    x_tr, _, _ = standardize(exp_x_tr)
    x_te, _, _ = standardize(exp_x_te)
    
    x_tr = add_bias_term(x_tr)
    x_te = add_bias_term(x_te)

    # Initialize some settings
    initial_w = np.zeros((x_tr.shape[1]))

    y_hat_te, w_opti, loss_mse = train_and_predict(y_tr, x_tr, y_te, x_te, model, seed, initial_w, max_iters, lambdas, gammas)

    # Compute the accuracy on the local test set
    accuracy_test = compute_accuracy(y_te, y_hat_te)
    f1_test = compute_f1_score(y_te, y_hat_te)
    
    print(f'Accuracy of {model} on the local test set : {accuracy_test:.4f}')
    print(f'F1-score of {model} on the local test set : {f1_test:.4f}')
    return accuracy_test, f1_test, w_opti