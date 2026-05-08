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

### Baseline Results

### Feature Selection Results

### Class Imbalance Results
