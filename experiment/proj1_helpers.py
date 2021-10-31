# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv

import numpy
import numpy as np
from experiment.cleaning import *
from pathlib import Path
from zipfile import ZipFile

from experiment.cleaning import remove_col_default_values, replace_by_default_value, check_all_azimuth_angles, clip_IQR
from experiment.expansion import add_bias_term, build_expansion, add_sin_cos, power_exp


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    
    # First check if the zip file was already unzipped
    if not Path(data_path).exists():
        print('==> Unzipping the data...')
        # If the data folder does not exist, then unzip it
        with ZipFile('data/data.zip', 'r') as zipped_file:
            # Extract the data files
            zipped_file.extractall(path='data')
            print('Data unzipped in the directory "data"')
    
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix        
    Parameters
    ----------
        weights :
            The weights computed in the train set
        data:
            the data to which we want to give predictions
    Returns
    -------
        y_pred : 
            The predictions
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def predict_labels_logistic_regression(weights, x):
    """
        Function that computes the predictions for the given data of the logistic regression model
        with the given weights
        
        Parameters 
        ----------
            weights :
                Trained weights of the model
            x :
                Data points
        Returns 
        -------
            Predicted labels of the model for the given data
    """
    y = 1 / (1 + np.exp(- x @ weights))

    y[y >= 0.5] = 1
    y[y < 0.5] = -1

    return y


def split_data(x, y, ratio, seed=1):
    """
        Split the given dataset into 2 different datasets (local train/test)
        according to the given ratio
        
        Parameters
        ----------
            x :
                Data points
            y :
                Outputs of the data points
            ratio :
                Ratio of samples to keep in the training set
            seed :
                Seed to initialize the RNG
        Returns 
        -------
            Tuple:
                - x_train
                - y_train
                - x_test
                - y_test
    """
    # set seed
    np.random.seed(seed)
    idx = np.random.choice(y.shape[0], int(ratio * y.shape[0]), replace=False)

    return x[idx], y[idx], np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)

#def expected_number_of_parameters(nb_features, max_degree, nb_angles_col):
    # If we do feature expansion before adding cosine
    #return 1+nb_features+(nb_features choose 2)+nb_angles_col+nb_features*(max_degree-1)
    # If we do feature expansion after adding cosine
    # return 1 + nb_features + (nb_features+nb_angles_col choose 2) + nb_features(
def process_test_set(test_data_path, col_removed_training, default_values_training, above_lim_training,
                     below_lim_training, means, stds, cols_angle, best_degree, expansion=True):
    """
        Load and pre-process test set
        Parameters
        ----------
            test_data_path :
                The os path to the test data set
             col_removed_training :
                Features removed in the training set
            default_values_training :
                Default values of features used in the training set
            above_lim_training :
                Upper limit of outliers used in the training set
            below_lim_training :
                Lower limit of outliers used in the training set
            means :
                Means used to standardise the features in the training set
            stds :
                Stds used to standardise the features in the training set
            cols_angle :
                Indexes of the features representing angles
            best_degree :
                Best degree to consider for the polynomial expansion
            expansion :
                Boolean, whether to use polynomial expansion or not
        Returns
        -------
             x_te_cleaned :
                 The test data set cleaned and ready for predictions
             ids_test :
                Ids of the test samples
            y_test :
                Labels of the test samples
    """
    # load the data
    y_test, x_test, ids_test = load_csv_data(test_data_path)

    # Apply pre-processing
    x_te_cleaned, _ = remove_col_default_values(x_test, cols_to_remove=col_removed_training)
    x_te_cleaned, _ = replace_by_default_value(x_te_cleaned, default_values_training)
    x_te_cleaned = check_all_azimuth_angles(x_te_cleaned, cols_angle)
    x_te_cleaned, _, _ = clip_IQR(x_te_cleaned, above_lim=above_lim_training, below_lim=below_lim_training)

    # Standardise the matrix and expand it
    x_te_cleaned = (x_te_cleaned - means) / stds

    x_te_cleaned = add_bias_term(x_te_cleaned)

    if expansion:
        # Need to increment the indexes of angle features since we added
        # the bias term
        x_te_cleaned = build_expansion(x_te_cleaned)
        x_te_cleaned = add_sin_cos(x_te_cleaned, np.array(cols_angle) + 1)
        x_te_cleaned = power_exp(x_te_cleaned, best_degree)

    return x_te_cleaned, ids_test, y_test