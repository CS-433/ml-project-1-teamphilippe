import numpy as np

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

def compute_accuracy(y, y_hat):
    """
        Compute the accuracy given the ground truth and the predicted data

        Parameters
        ----------
            y :
                Ground labels
            y_hat :
                Predicted labels
        Returns
        -------
            Accuracy in the range [0;1]
    """
    return np.mean(np.equal(y, y_hat))

def compute_f1_score(y, y_hat):
    """
        Compute the accuracy given the ground truth and the predicted data according to the definition on Wikipedia.w

        Parameters
        ----------
            y :
                Ground labels
            y_hat :
                Predicted labels
        Returns
        -------
            F1-score in the range [0;1]
    """
    not_classified_correctly = np.sum(~np.equal(y, y_hat))
    mask_classified_positive = y>0
    
    subs_y = y[mask_classified_positive]
    subs_y_hat = y_hat[mask_classified_positive]
    
    tp = np.equal(subs_y,subs_y_hat).sum()
    
    return tp/(tp+1/2*not_classified_correctly)


def compute_rmse(mse):
    """
        Compute the Root Mean Square Error (RMSE) given the Mean Square Error (MSE).
        
    Parameters
    ----------
        mse :
            Mean Square Error
    Returns
    -------
        rmse : 
            Root Mean Square Error
    """
    return np.sqrt(2 * mse)