## Stroke Prediction with Class Imbalance: Model Comparison and Handling Strategies
Nathan Derhake, Chanbin Lim, Serene Saad, Emily Weiss

CSCI 567 Project 


## Dataset Source
The dataset used for this project is the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), authored by kaggle user [fedesoriano](https://www.kaggle.com/fedesoriano). 

This dataset has been made public for use for educational purposes. 



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

#### RFE

To produce the results discussed in the RFE section, we run the python notebook: ```rfe_presentation.ipynb```

#### Importance Based Feature Selection Per Model

To produce the results discussed in this section, we run the python notebook ```feature_selection_notebook.ipynb```, and the file ```feature_selection_by_importance.py```.


### Class Imbalance Results

##### Random Oversampling

To produce the results discussed in the Random Oversampling section, we run the cells up to the "Logistic Regression" header in the ```random_oversampling/random_oversampling.ipynb``` notebook.
This produces the train and test results for the models using the same hyperparameters as the baselines, but with a randomly oversampled training set. 

##### SMOTE

To produce the results discussed in the SMOTE section, we run the python file ```smote/smote_baselines.py```.

##### Class Weights

To produce the results discussed in the Class Weights section, we run the cells underneath the "Final Results for Writeup" section in the ```class weights imbalance method/class_weights.ipynb``` file. 

##### Combinations of Approaches

To produce the class weights + SMOTE results in the writeup, we run the python file ```smote/weight_and_smote_combi.py```.

To produce the class weights + random oversampling results in the writeup, we run the cells up to the "Logistic Regression" header in ```cw_ro/weighted_random_oversampling.ipynb```.  


##### Artificial Minority Neighbors

To produce the results discussed in the Artificial Minority Neighbors section, we run the cells up to the "New hyperparameters for neighbors" header in the ```neighbors/neighbors.ipynb``` notebook.
This produces the train and test results for the models using the same hyperparameters as the baselines, but with a randomly oversampled training set. 

##### Calibration

For generating all calibration results and figures:

```c:/[FILEPATH]/Stroke-Prediction-ML/venv/Scripts/python.exe c:/Users/natha/github/Stroke-Prediction-ML/src/calibration/calibration_experiments.py```

For testing functions in ```calibration.py```:

```c:/[FILEPATH]/Stroke-Prediction-ML/venv/Scripts/python.exe c:/Users/natha/github/Stroke-Prediction-ML/src/calibration/calibration.py```
