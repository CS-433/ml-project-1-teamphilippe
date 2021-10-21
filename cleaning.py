# -*- coding: utf-8 -*-

import numpy as np


# Check if the values of azimuthal columns are in the range [-pi, pi[
def check_azimuth_and_rerange(x, col_idx):
    mask = (x[:, col_idx] > np.pi) | (x[:, col_idx] < -np.pi)
    print(f'Number of values outside [-pi;pi[ (col {col_idx}): {np.sum(mask)}')
    
    # Set the non valid azimuths back in the [-pi;pi[ interval
    x[mask, col_idx] = (x[mask, col_idx] % (2 * np.pi)) - np.pi
    return x

# We remove the features where more than half of the rows are -999
def remove_col_default_values(x):
    cols_to_remove = []
    for i in range(x.shape[1]) :
        if (np.where(x[:, i] == -999)[0]).shape[0] > (x.shape[0]/2) :
            cols_to_remove.append(i)

    return np.delete(x, cols_to_remove, axis=1)

def check_nb_rows_default_features(x):    
    def check_row_default_features(row):
        return np.sum(row == -999) > (row.shape[0]/2)
    
    mask = np.apply_along_axis(check_row_default_features, axis=0, arr=x)
    print(f'Number of rows : {np.sum(mask)}')
    
    

def standardize(x):
    """
        Standardize the original data set according to the formula : (x-mean(x))/std(x).
        
    Parameters
    ----------
        x :
            Data points to standardize
    Returns
    -------
        x : 
            Data points standardized
        mean_x :
            Mean of x before standardization
        std_x :
            Standard deviation of x before standardization
        
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

