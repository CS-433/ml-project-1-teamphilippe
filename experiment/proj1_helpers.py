# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from cleaning import *
from pathlib import Path
from zipfile import ZipFile


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    
    # First check if the zip file was already unzipped
    if not Path(data_path).exists():
        print('==> Unzipping the data...')
        # If the data folder does not exist, then unzip it
        with ZipFile('../data/data.zip', 'r') as zipped_file:
            # Extract the data files
            zipped_file.extractall(path='../data')
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
