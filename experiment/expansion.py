import numpy as np
import itertools


def build_expansion(data):
    """
    Build the a data matrix composed of product of every pair of columns
    Parameters
    ----------
        data :
            The data matrix
    Returns
    -------
        all_columns_com : 
            A new dataset composed of product of every pair of columns
    """
    # Build all possible pairs of number up to n=data.shape[1]
    all_combinations = list(itertools.combinations(range(data.shape[1]),2))
    # Explicitly add the (0,0) terms as we always want a bias term.
    all_combinations.append((0,0))
    # Sort the combinations in terms of the first and then second element of the tuple
    combinations = np.array(sorted(all_combinations))
    
    # Construct the product of every pair of columns in the dataset
    all_columns_com = data[:, combinations[:, 0]] * data[:, combinations[:, 1]]
    return all_columns_com


def add_sin_cos(data, columns):
    """
        Augment the collection of features with combination of sin and cos functions applied to all the features
        
        Parameters
        ----------
            data :
                Data matrix
            columns :
                List of the indexes of the columns to use to in the sin and cos expansion
        
        Return
        ------
            New data matrix augmented with new combinations of features
    """
    data = np.concatenate((data, np.cos(data[:, columns]), np.sin(data[:, columns])), axis=1)
    return data


def power_exp(data, max_degree):
    """
        Power expansion of the data matrix passed as argument
        
        Parameters
        ----------
            data :
                Data matrix
            max_degree :
                Maximum degree to use in the power expansion
            base_cols :
                List of the indexes of the columns to use in the power expansion
        Return
        ------
            New augmented data matrix
    """
    base_cols = list(range(1, 24))
    for i in range(2, max_degree + 1):
        base_cols_powered = data[:, base_cols] ** i
        data = np.concatenate((data, base_cols_powered), axis=1)

    return data


def add_bias_term(data):
    """
    Add a column of 1 in the data matrix to handle the bias term
    Parameters
    ----------
        data :
            The data 
    Returns
    -------
        data_with_bias : 
            The dataset with a column of 1 
    
    """
    return np.hstack((np.ones((data.shape[0], 1)), data))
