from proj1_helpers import *

"""
Script that
    - loads the training data
    - train the model with the best hyperparameters found during the cross-validation (done in the notebook "project1")
    - make predictions on the test set
    - save the submission csv file "submission.csv".
"""


def load_and_preprocess_training_data(best_degree, cols_angle):
    training_data_path = "data/train.csv"
    
    # Load the training data
    y, x, ids = load_csv_data(data_path, sub_sample=False)
    
    # Remove the features where more than half of the rows are -999
    x_cleaned, col_removed_training = remove_col_default_values(x)
    
    # Replace default values
    x_cleaned, default_values_training = replace_by_default_value(x_cleaned)
    
    # Check and correct the angles features
    x_cleaned = check_all_azimuth_angles(x_cleaned, cols_angle)
    
    # Remove outliers
    x_cleaned, above_lim_training, below_lim_training = clip_IQR(x_cleaned)
    
    # Standardize the data
    x_cleaned, means, stds = standardize(x_cleaned)
    
    # Expand the features
    x_cleaned = add_bias_term(x_cleaned)
    x_cleaned = add_sin_cos(x_cleaned, cols_angle)
    x_cleaned = build_expansion(x_cleaned)
    
    # Polynomial expansion with the best degree found
    x_cleaned = power_exp(x_cleaned, best_degree)
    
    return x_cleaned, y, means, stds, col_removed_training, default_values_training, above_lim_training, below_lim_training


def load_and_preprocess_test_data(col_removed_training, default_values_training, above_lim_training, below_lim_training, means, stds, cols_angle, best_degree):
    
    test_path = 'data/test.csv'
    
    return process_test_set(test_path, col_removed_training, default_values_training, above_lim_training, below_lim_training, means, stds, cols_angle, best_degree)


def main():
    # Indexes of features representing angles
    cols_angle = [11, 14, 16, 21]
    
    # Best hyperparameters found (lambda and degree
    # of the polynomials expansion) during the cross validation done
    # in the notebook "project1".
    best_lambda = None
    best_degree = None
    
    print('Loading and preprocessing training data...')
    # Load and preprocess training data
    x_tr, y_tr, means, stds, col_removed_training, default_values_training, above_lim_training, below_lim_training = load_and_preprocess_training_data(best_degree, cols_angle)
    
    print('Training model...')
    # Train the model
    w, loss_mse = ridge_regression(y_tr, x_tr, best_lambda)
    
    print('Loading and preprocessing test data...')
    # Load and preprocess test set
    x_te_cleaned, ids_test, y_test = load_and_preprocess_test_data(col_removed_training, default_values_training, above_lim_training, below_lim_training, means, stds, cols_angle, best_degree)
    
    print('Predicting labels for the test set...')
    # Make predictions for the test set
    y_test_pred = predict_labels(w_best, x_te_cleaned)
    
    print('Creating submission file...')
    # Save the submission file
    create_csv_submission(ids_test, y_test_pred, 'submission.csv')
    
    print(f'Submission files saved.')
    
if __name__ == '__main__':
    main()