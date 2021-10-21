# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from cleaning import *
import itertools


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def build_expansion(data):
    combinations = np.array(list(itertools.combinations_with_replacement(range(data.shape[1]),2)))
    return data[:,combinations[:,0]] *data[:,combinations[:,1]]

def add_bias_term(data):
    return np.hstack((np.ones((data.shape[0],1)),data))
    


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
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
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
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