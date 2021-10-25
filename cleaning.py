# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *

def check_all_azimuth_angles(x, cols_angle):
    """
    This function check that all the angle valued columns are in the range [-pi,pi[
    
    Parameters
    ----------
        x : 
            The dataset to which we want to apply the method
    Returns
    -------
        x : 
            The cleaned dataset 
    """
    for col in cols_angle:
        x = check_azimuth_and_rerange(x, col)
    
    return x

# Replace the remaining -999 by the median/mean/0s
def replace_by_default_value(x, default_values = None):
    """
    Replace the reamining -999 by the a default value
    
    Parameters
    ----------
        x : 
            The dataset to which we want to apply the method
        default_values: 
            The default value for each column in the dataset
    Returns
    -------
        x : 
            The dataset with -999 replaced by a default
    """
    if(default_values is None):
        #If no default values specified (i.e. we clean the training set), compute the mean for each column and replace -999 by this value
        
        default_values = []
        for i in range(x.shape[1]):
            mask = x[:, i] == -999
            median = np.median(x[~mask, i]) # or 0
            x[mask, i] = median
            # Store the value so that it can be used to replace -999 in the test set 
            default_values.append(median)
    else:
        # We clean the test set
        for i in range(x.shape[1]):
            # Replace -999 by the median computed in the training set
            mask = x[:, i] == -999
            x[mask, i] = default_values[i]
    return x, default_values
            
def percentile_comparison(row, above_lim, below_lim) :
    """
    Clip row value to the above or below percentiles
    Parameters
    ----------
        row : 
            The row we want to modify
        above_lim: 
            List of above limits for every column
        below_lim: 
            List of below limits for every column
    Returns
    -------
        row : 
            The clipped row 
    """
    for idx, percs in enumerate(zip(above_lim, below_lim)) :
        # If the row value is above the limit, clip it to the max percentile value retained
        if row[idx] > percs[0] :
            row[idx] = percs[0]
        # If the row value is below the limit, clip it to the min percentile value retained  
        if row[idx] < percs[1] :
            row[idx] = percs[1]
    return row

def clip_IQR(x, k=1.5, percentiles=[25, 75], above_lim = None, below_lim = None):
    """
    Clip value of the dataset using interquartile range method
    Parameters
    ----------
        x : 
            The dataset we want to clip
        k: 
            Constant value to choose which type of outliers we want to remove
            Typical value for k: 
                - 1.5 : remove outlier
                - 3.0 : remove extreme outliers
                - 6.0 : Remove very extreme outliers
        percentiles: 
            Above and below percentiles to clip the value
        above_lim: 
            List of above limits for every column
        below_lim:
            List of below limits for every column
    Returns
    -------
        row : 
            The clipped dataset 
    """
    if(above_lim is None or below_lim is None):
        # We are processing the training set
        # Compute the percentiles
        perc = np.percentile(x, percentiles, axis=0)
        
        # Compute interequartile range and the limits
        iqr = (perc[1,:]-perc[0,:])
        above_lim = perc[1,:] + k * iqr
        below_lim = perc[0,:] - k * iqr
    
    # Clip the value in every row 
    x = np.apply_along_axis(lambda row : percentile_comparison(row, above_lim, below_lim),
                           1, x)
    
    return x, above_lim, below_lim

# 
def check_azimuth_and_rerange(x, col_idx):
    """
    Check and clip if the values of azimuthal columns are in the range [-pi, pi[
    Parameters
    ----------
        x : 
            The dataset from which we want to check the angle vlaues
        col_idx:
            The column that needs to be check
    Returns
    -------
        x : 
            The checked dataset 
    """
    # Check if the value is in the range
    mask = (x[:, col_idx] >= np.pi) | (x[:, col_idx] < -np.pi)
    print(f'Number of values outside [-pi;pi[ (col {col_idx}): {np.sum(mask)}')
    
    # Set the non valid azimuths back in the [-pi;pi[ interval
    x[mask, col_idx] = (x[mask, col_idx] % (2 * np.pi)) - np.pi
    return x

def remove_col_default_values(x, cols_to_remove = None):
    """
    Remove the features where more than half of the rows are -999
    Parameters
    ----------
        x : 
            The dataset from which we want to remove columns
        cols_to_remove:
            The column number we want to remove or None if we need to compute which columns should be removed. 
    Returns
    -------
        row : 
            The clipped dataset 
    """
    if cols_to_remove is None:
        # We are processing the training set 
        # Compute the row in which more than the half is set to -999
        cols_to_remove = []
        half_nb = (x.shape[0]/2)
        for i in range(x.shape[1]) :
            if (np.where(x[:, i] == -999)[0]).shape[0] > half_nb :
                cols_to_remove.append(i)
    
    return np.delete(x, cols_to_remove, axis=1), cols_to_remove

def check_nb_rows_default_features(x):
    """
    Compute and print the number of rows that have -999 in more than half the features
    Parameters
    ----------
        x : 
            The dataset from which we want to compute the count
    """
    def check_row_default_features(row):
        # Return if the current row has -999 in more than half the features
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
   