## Stroke Prediction with Class Imbalance: Model Comparison and Handling Strategies
Nathan Derhake, Chanbin Lim, Serene Saad, Emily Weiss

CSCI 567 Project 


## Dataset Source
The dataset used for this project is the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), authored by kaggle user [fedesoriano](https://www.kaggle.com/fedesoriano). 

This dataset has been made public for use for educational purposes. 

## Repository structure

stroke_data.py:
- general file used to access the train/test or train/test/val splits of the preprocessed data

preprocessing.ipynb:
- python notebook used to preprocess the data, and produce the various files contained in the data directory

data: 
- Contains various versions of the preprocessed stroke prediciton dataset
- Version used for majority of experiments is: knn-standardize-distance.csv
- Version used for calibration experiments is: **TODO**

src/baseline_models:
- Contains the code to do gridsearch to find the best parameters for our baseline models
- Contains the code to train and test the baseline models used for our experiment

src/feature_selection:
- **todo**



## Commands to generate results 
**todo**

### Installing Packages
run ```pip install -r requirements.txt```


### Baseline Results
To produce the results discussed in the Baseline Models section, we run the cells in the ```baseline/baseline.ipynb``` notebook under the header: "Run baselines for writeup". 
This produces train and test results for the baseline models with the chosen hyperparameters via Grid Search, as shown earlier in the file. 

### Feature Selection Results

#### Feature Importance

To produce the results discussed in the Random Forest Feature Importance section, we run the python notebook: ```importance_rf_presentation.ipynb```


### Class Imbalance Results

##### Random Oversampling

To produce the results discussed in the Random Oversampling section, we run the cells up to the "Logistic Regression" header in the ```random_oversampling/random_oversampling.ipynb``` notebook.
This produces the train and test results for the models using the same hyperparameters as the baselines, but with a randomly oversampled training set. 

##### Artificial Minority Neighbors

To produce the results discussed in the Artificial Minority Neighbors section, we run the cells up to the "New hyperparameters for neighbors" header in the ```neighbors/neighbors.ipynb``` notebook.
This produces the train and test results for the models using the same hyperparameters as the baselines, but with a randomly oversampled training set. 

##### Calibration

For generating all calibration results and figures:

```c:/[FILEPATH]/Stroke-Prediction-ML/venv/Scripts/python.exe c:/Users/natha/github/Stroke-Prediction-ML/src/calibration/calibration_experiments.py```

For testing functions in ```calibration.py```:

```c:/[FILEPATH]/Stroke-Prediction-ML/venv/Scripts/python.exe c:/Users/natha/github/Stroke-Prediction-ML/src/calibration/calibration.py```
