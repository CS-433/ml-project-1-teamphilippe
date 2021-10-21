# -*- coding: utf-8 -*-

import numpy as np

def check_all_azimuth_angles(x):
    x_cleaned = x.copy()
    x_cleaned = check_azimuth_and_rerange(x_cleaned, 15)
    x_cleaned = check_azimuth_and_rerange(x_cleaned, 18)
    x_cleaned = check_azimuth_and_rerange(x_cleaned, 20)
    x_cleaned = check_azimuth_and_rerange(x_cleaned, 25)
    x_cleaned = check_azimuth_and_rerange(x_cleaned, 28)
    return x_cleaned

# Replace the remaining -999 by the median/mean/0s
def replace_by_default_value(x_cleaned, default_values = None):
    if(default_values is None):
        default_values = []
        for i in range(x_cleaned.shape[1]):
            mask = x_cleaned[:, i] == -999
            median = np.median(x_cleaned[~mask, i]) # or 0
            x_cleaned[mask, i] = median
            default_values.append(median)
    else:
        for i in range(x_cleaned.shape[1]):
            mask = x_cleaned[:, i] == -999
            x_cleaned[mask, i] = default_values[i]
    return x_cleaned, default_values
            
def percentile_comparison(row, above_lim, below_lim) :
    for idx, percs in enumerate(zip(above_lim, below_lim)) :
        if row[idx] > percs[0] :
            row[idx] = percs[0]
        if row[idx] < percs[1] :
            row[idx] = percs[1]
    return row

def clip_IQR(x, k=1.5, percentiles=[25, 75], above_lim = None, below_lim = None):
    if(above_lim is None or below_lim is None):
        perc = np.percentile(x, percentiles, axis=0)

        iqr = (perc[1,:]-perc[0,:])
        above_lim = perc[1,:] + k * iqr
        below_lim = perc[0,:] - k * iqr
    
    x = np.apply_along_axis(lambda row : percentile_comparison(row, above_lim, below_lim),
                           1, x)
    
    return x, above_lim, below_lim

# Check if the values of azimuthal columns are in the range [-pi, pi[
def check_azimuth_and_rerange(x, col_idx):
    mask = (x[:, col_idx] > np.pi) | (x[:, col_idx] < -np.pi)
    print(f'Number of values outside [-pi;pi[ (col {col_idx}): {np.sum(mask)}')
    
    # Set the non valid azimuths back in the [-pi;pi[ interval
    x[mask, col_idx] = (x[mask, col_idx] % (2 * np.pi)) - np.pi
    return x

# We remove the features where more than half of the rows are -999
def remove_col_default_values(x, cols_to_remove = None):

    if cols_to_remove==None:
        cols_to_remove = []
        for i in range(x.shape[1]) :
            if (np.where(x[:, i] == -999)[0]).shape[0] > (x.shape[0]/2) :
                cols_to_remove.append(i)

    return np.delete(x, cols_to_remove, axis=1), cols_to_remove

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

