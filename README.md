# Machine Learning - Project 1 - TeamPhilippe

### Team Members
- Eloïse Doyard (272300)
- Alessio Verardo (282634)
- Cyrille Pittet (282445)


### Structure of the project
We separated the code into multiple files instead of having all the implementations in a single huge unreadable file.
As asked, the implementations.py file contains all the functions for the models.

- implementations.py : models we were asked to implement, the SGD function we used to train the models
- project1.ipynb : notebook containing the data analysis, cleaning and features processing steps as
  well as the cross validation for the hyperparameters
- run.py : script to reproduce our results
- data :
    - data.zip : file containing the training and test data
- experiment :
    - cleaning.py : all the functions used in the cleaning of the dataset
    - cross_validation.py : functions used for the cross validation
    - expansion.py : functions used for features processing and expansion
    - metrics.py : functions computing losses, accuracy, F1 score
    - proj1_helpers.py : functions to load the dataset, create the submission csv file, split the dataset into local training/test sets, predict labels
    - simulation.py : utility functions/pipelines to train the models, run cross validation, predict test labels and compute accuracies
    - visualisation.py : utility functions for different kind of plots
- models :
  - loss_gradient.py : definitions of the losses and gradients of the different models considered 
  

### How to run the code
We provide a run.py file which does the following :
1. Unzip and load the data
2. Clean and preprocess the training data
3. Train the ridge regression model with the best hyperparameters found in the cross validations we did in the notebook
4. Load the test data and apply the same preprocessing on them
5. Predict the labels for the test data and save them as a csv file called 'submission.csv'

Note : as explained in the report, we noticed a bug in our code after making our best submission on AICrowd.
Therefore, this run.py script reproduces the submission with an accuracy of 82.8% on AICrowd and not the actual best submission that we made.

To run this code, you will need Python 3 and numpy installed. To execute it you can :
- Simply run it in a terminal : ```python3 run.py```
- Or call the main function from within a notebook : 
```python
from run import main
main()
```