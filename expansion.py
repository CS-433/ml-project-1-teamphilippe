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
    combinations = np.array(list(itertools.combinations_with_replacement(range(data.shape[1]),2)))
    # Construct the product of every pair of columns in the dataset
    all_columns_com = data[:,combinations[:,0]] *data[:,combinations[:,1]]
    return all_columns_com

def add_sin_cos(data, columns):
    cols = columns.copy()
    cols.append(0)
    comb = list(itertools.combinations_with_replacement(cols, 2))
    comb.remove((0,0))
    
    combinations = np.array(comb)
    # Construct the product of every pair of columns in the dataset
    all_columns_com = np.cos(data[:,combinations[:,0]]) *np.sin(data[:,combinations[:,1]])
    
    data = np.concatenate((data, all_columns_com), axis=1)
    return data

def power_exp(data, max_degree, base_cols=list(range(1,24))):
    for i in range(1,max_degree+1):
        base_cols_powered = data[:,base_cols]**i
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
    data_with_bias = np.hstack((np.ones((data.shape[0],1)),data))
    return data_with_bias