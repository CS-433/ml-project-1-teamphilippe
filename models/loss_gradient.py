import numpy as np

"""
Logistic regression related functions
"""


# Logistic regression
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
    sigm = sigmoid(tx @ w)
    return tx.T @ (sigm - y)


def compute_loss_logistic_regression(y, tx, w, *args):
    """
        Function that compute the loss of the weight vector w in the case of the logistic regression
        Note that the loss is the average loss over the whole dataset to allow comparison between datasets
        with different number of items.
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
    return  np.sum(np.log(1 + np.exp(xtw)) - y * xtw) / tx.shape[0]


# Regularized logistic regression
def compute_loss_reg_logistic_regression(y, tx, w, lambda_):
    """
        Function that compute the loss of the weight vector w in the case of the regularised logistic regression
        
        Parameters
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the loss
            lambda_ :
                Regulariser of this regularized logistic regression
        Returns 
        -------
            Loss of the weight vector, using the regularised logistic regression method, computed with the parameters
    """
    return compute_loss_logistic_regression(y, tx, w)


def compute_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """
        Function that computes the gradient of the weight vector w in the case of the regularised logistic regression
        
        Parameters 
        ----------
            y :
                Outputs of the data points
            tx :
                Data points
            w :
                Potential weight vector used to compute the gradient
            lambda_ :
                Regulariser of this regularized logistic regression
        Returns 
        -------
            Gradient, using the regularised logistic regression method, computed with the parameters
            
    """
    return compute_gradient_logistic_regression(y, tx, w) + 2 * lambda_ * w


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
                Regulariser of this ridge regression
        Returns 
        -------
            Gradient of the weight vector, using the ridge regression method, computed with the parameters
    """
    return compute_gradient_least_squares(y, tx, w) + 2 * lambda_ * w


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
                Regulariser of this ridge regression
        Returns 
        -------
            Loss of the weight vector, using the ridge regression method, computed with the parameters
    """
    return compute_loss_least_squares(y, tx, w)


def sigmoid(x):
    """
    Apply the sigmoid function to the input x
    Parameters
        ----------
            x:
                The vectors to which we want to apply the sigmoid function
        Returns
        -------
            The sigmoid of the input
    """
    return 1.0 / (1 + np.exp(-x))