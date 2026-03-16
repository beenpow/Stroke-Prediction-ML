# Stroke Prediction with Class Imbalance: Model Comparison and Handling Strategies

Requirements:   
What research question are you trying to answer?

* **Primary question:** Which features and ML algorithms best predict stroke occurrence in this dataset?
* **Secondary question:** Can we improve prediction by addressing class imbalance?
  * The dataset is highly imbalanced: approximately 95% of records are no-stroke and 5% are stroke.
  * How effective are existing methods for handling class imbalance (e.g., resampling, class weights)?
  * Can we propose or adapt methods to better deal with this imbalance?

Why is this question interesting to you?

* **Interest:** We are drawn to exploring class imbalance and the predictive power of different models on a dataset in the medical field.
* **Motivation:** Real world data is messy and often has extreme cases of class imbalance, which could result in very high accuracy scores but low F1. Being able to gracefully handle class imbalance to produce accurate predictions is especially important in high-stakes situations such as predicting whether a patient is likely to have a stroke. False negatives could prevent patients from seeking preventative care or making the necessary lifestyle changes to minimize the risk of stroke. 

What kind of data are you collecting or what datasets will you use?

* We use the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data). The data has 10 features: gender, age, hypertension, heart disease, marriage status, work type, residence type, avg. glucose level, BMI, and smoking status—used to predict stroke (target provided in the dataset). Seven are categorical and three are numerical.
* The data passes basic sanity checks, which indicate that it is likely real rather than synthetically generated (e.g., no young children with married/smoking/non-child work-type entries). There are some missing BMI values, consistent with physically collected records.
* Kaggle states the data comes from patient records. The dataset was created by [fedesoriano](https://www.kaggle.com/fedesoriano), a Kaggle Datasets Grandmaster, which we take as an indication of reliability and legitimacy.

What algorithms will you try?

* We will try: Random Forest; neural networks with regularization ([scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)); Logistic regression; SVM; [XGBoost](https://xgboost.readthedocs.io/en/stable/).

What experiments and analysis will you run?

* We will evaluate predictive performance (e.g., accuracy, F1, recall) for binary stroke vs no-stroke prediction.
* **Planned analyses:**
  * **Class imbalance:** Compare resampling (oversampling/undersampling), class weights, etc., vs baseline; report recall and F1. We will refer to and build on [this analysis of data imbalance](https://hemostasistoday.com/insight/favour-kpokpe-32544).
  * **Feature importance:** XGBoost, ablation; optionally $l_0$ regularization.
  * **Model comparison:** Compare all models via hold-out or cross-validation (accuracy, F1, recall).
  * **Overfitting:** Use train / validation / test split and monitor generalization.
  * **Categorical and numerical features:** One-hot encoding; concatenate features (e.g., age, bmi, is_Male, is_Female, …).
  * **Missing BMI:** Compare dropping missing values vs imputation (e.g., from observed data).

What do you plan to finish by the pre-final report and check-in? (Check in April 20th)

* **Experiments:** By the week of April 20th, we will have completed experiments and analysis for at least two of our algorithms (Random Forest, neural networks, Logistic regression, SVM, XGBoost) and made substantial progress on the third, with time set aside for debugging and sound experimental design.
* **Reporting:** We will be able to report on each model’s effectiveness for stroke prediction and on how well our predictors handle the unbalanced dataset (e.g., recall, F1). We will have a clear picture of our results and assumptions.
* **Next steps:** We will wrap up all experiments, begin the project report, and decide how to present our results as a coherent story.

