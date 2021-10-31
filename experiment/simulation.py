from experiment.cross_validation import *
from experiment.metrics import *
from experiment.proj1_helpers import *

"""
Customise version of the stochastic gradient descent to show plot of the loss function 
"""


def stochastic_gradient_descent_validation(y, tx, y_te, x_te, max_iters, gamma, compute_loss, compute_gradient,
                                           lambda_=0, batch_size=1):
    """
        Function that run the stochastic gradient descent and output the best weight vector and its according loss.
        Plots the loss over the iterations of the SGD.
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            y_te:
                Outputs of test data points
            x_te:
                Test data points
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            gamma :
                Step size of the stochastic gradient descent
            compute_loss :
                Function to use to compute the loss of the weight vector
            compute_gradient:
                Function to use to compute the gradient of the weight vector
            lambda_ :
                The importance of the regulariser
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

    w = np.zeros(tx.shape[1])

    # start of the stochastic gradient descent
    for n_iter in range(max_iters):
        # start of the compute of the stochastic gradient for the weight vector candidate
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # stochastic gradient with particular batch of data points and its corresponding outputs
            sgrad = compute_gradient(y_, tx_, w, lambda_)

            # Gradient Descent
            w = w - gamma * sgrad
            # compute the loss of the corresponding optimal weight vector on the training and test sets
            loss = compute_loss(y, tx, w, lambda_)
            loss_te = compute_loss(y_te, x_te, w, lambda_)

            # store losses for both test and train set
            losses.append(loss)
            losses_test.append(loss_te)
        
    # Plot the test and training losses to see the convergence
    plt.plot(losses)
    plt.plot(losses_test)
    plt.show()
    return w, losses[-1]


def train_and_predict(y_tr, x_tr, y_te, x_te, model, seed, max_iters, lambdas, max_degree, gamma):
    """
        Train the given model and predict the labels of the local test set.
        If necessary for the model, performs a cross validation on the hyperparameters.

        Parameters
        ----------
            y_tr :
                Outputs of the training data points
            x_tr :
                Training data points
            y_te:
                Outputs of the test data points
            x_te :
                Test data points
            model :
                String, model to use
            seed :
                Seed to initialize the RNG
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            lambdas :
                The different coefficients to try in the cross validation
            max_degree:
                Max degree to consider for the polynomial expansion
            gamma :
                Learning rate
        Returns
        -------
        Tuple :
                - Predicted labels on the given local test set
                - Optimal weights found
                - Loss on the test set
                - The best degree for power expansion
    """
    # Default value for regulariser coefficient if not used
    best_lambda = 0

    if model in ['logistic_regression', 'reg_logistic_regression']:
        if model == 'logistic_regression':
            # Lambdas = [0.0] as we do not need to find lambda (not used in logistic regression)
            best_lambda, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_logistic_regression, compute_gradient_logistic_regression,
                max_iters, lambdas=[0.0], max_degree=max_degree, gamma=gamma, seed=seed)

            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, max_iters, gamma,
                                                                 compute_loss_logistic_regression,
                                                                 compute_gradient_logistic_regression)

        elif model == 'reg_logistic_regression':
            # Perform cross validation to find the best regulariser and polynomial degree
            best_lambda, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_reg_logistic_regression, compute_gradient_reg_logistic_regression,
                max_iters, lambdas=lambdas, max_degree=max_degree, gamma=gamma, seed=seed)

            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, max_iters, gamma,
                                                                 compute_loss_reg_logistic_regression,
                                                                 compute_gradient_reg_logistic_regression,
                                                                 lambda_=best_lambda)
        
        # Predict the labels on the local test set
        y_hat_te = predict_labels_logistic_regression(w, x_te)
    else:
        if model == 'least_squares_GD':
            # Full gradient descent is equivalent to SGD with
            # batch size = N, i.e. the size of the training set

            # Cross validation only to find a good learning rate gamma (lambda is not used in least_squares GD)
            best_lambda, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, lambdas=[0.0], max_degree=max_degree, gamma=gamma, seed=seed, batch_size=y_tr.shape[0])

            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Train the model with the best parameters on the local training set + plot the loss curves.
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, max_iters, gamma,
                                                                 compute_loss_least_squares,
                                                                 compute_gradient_least_squares, batch_size=len(x_tr))

        elif model == 'least_squares_SGD':
            # Cross validation only to find a good learning rate gamma (lambda is not used in least_squares GD)
            best_lambda, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares,
                max_iters, lambdas=[0.0], max_degree=max_degree, gamma=gamma, seed=seed)

            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Train the model with the best parameters on the local training set + plot the loss curves
            w, loss_mse = stochastic_gradient_descent_validation(y_tr, x_tr, y_te, x_te, max_iters, gamma,
                                                                 compute_loss_least_squares,
                                                                 compute_gradient_least_squares)

        elif model == 'least_squares':
            # Cross validation only to find a good learning rate gamma (lambda is not used in least_squares)
            _, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_least_squares, compute_gradient_least_squares, 
                max_iters, lambdas=[0.0], max_degree=max_degree, gamma=gamma, seed=seed, optimization='least_squares')
            
            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Find the least squares solution via the normal equation
            w, loss_mse = least_squares(y_tr, x_tr)

        elif model == 'ridge_regression':
            # Cross validation to find the best lambda (no gradient descent => gamma is not used)
            best_lambda, best_degree = perform_cross_validation(
                y_tr, x_tr,
                compute_loss_ridge, compute_gradient_ridge_regression,
                max_iters, lambdas, max_degree,
                seed=seed, optimization='ridge_normal_eq')

            x_tr = power_exp(x_tr, best_degree)
            x_te = power_exp(x_te, best_degree)

            # Return the solution to the (regularised) normal equation
            w, loss_mse = ridge_regression(y_tr, x_tr, best_lambda)

        else:
            print(f'Model ({model}) not supported')
            return
                
        # Predict the labels on the local test set
        y_hat_te = predict_labels(w, x_te)

    return y_hat_te, w, loss_mse, best_degree, best_lambda


def run_experiment(y, x, model, seed, ratio_split_tr, cols_angle, max_iters=100, lambdas=np.logspace(-15, 0, 15),
                   gammas=0.0001, max_degree=9):
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
            cols_angle:
                The column number of angle features
            max_iters :
                Maximum number of iteration of the stochastic gradient descent
            lambdas :
                The different coefficients to try in the cross validation
            gammas :
                The different learning rates to try in the cross validation
            max_degree:
                The maximal degree to which we should try polynomial expansion
        Returns
        -------
        Tuple :
            - The accuracy of the model on the local test set
            - F1 score of the model on the test set
            - Optimal weights found
            - The best degree for polynomial expansion
    """
    # Split the training set into a local training set and a local test set
    x_tr, y_tr, x_te, y_te = split_data(x, y, ratio=ratio_split_tr, seed=seed)

    # Standardise all the features
    x_tr, means, stds = standardise(x_tr)
    x_te = (x_te - means) / stds

    x_tr = add_bias_term(x_tr)
    x_te = add_bias_term(x_te)

    if model in ['logistic_regression', 'reg_logistic_regression']:
        # As explained on the forum, the input for the logistic regression should have label in {0,1}
        y_tr[y_tr == -1.0] = 0.0   
        
    x_tr = add_sin_cos(x_tr, np.array(cols_angle) + 1)
    x_te = add_sin_cos(x_te, np.array(cols_angle) + 1)
    x_tr = build_expansion(x_tr)
    x_te = build_expansion(x_te)


    print("End of processing + expansion")
    print("Beginning training")

    y_hat_te, w_opti, loss_mse, best_degree, best_lambda = train_and_predict(y_tr, x_tr, y_te, x_te,
                                                                             model, seed, max_iters, lambdas,
                                                                             max_degree, gammas)

    # Compute the accuracy on the local test set
    accuracy_test = compute_accuracy(y_te, y_hat_te)
    f1_test = compute_f1_score(y_te, y_hat_te)

    print(f'Accuracy of {model} on the local test set : {accuracy_test:.4f}')
    print(f'F1-score of {model} on the local test set : {f1_test:.4f}')
    return accuracy_test, f1_test, w_opti, best_degree, best_lambda
