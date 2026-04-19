import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from src.stroke_data import get_stroke_data_for_cv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# X_train, X_test, y_train, and y_test are defined here.
from src.baseline_models.custom_gridsearch import *
from src.calibration import *

## Using Scikit learn Multi-layer perceptron (MLP)
## https://scikit-learn.org/stable/modules/neural_networks_supervised.html

params = {"alpha": [1.0],
            "solver": ["adam"],
            "max_iter": [200],
            "hidden_layer_sizes": [(100, 24)],
            "activation": ["relu"],
}

best_params, score = mlp_fake_grid_search(params, f1)

print("The best parameters are " + str(best_params) + " with score of " + str(score))

# # Define the model
clf = MLPClassifier(random_state=1, **best_params)

# Fit the model
clf.fit(X_train, y_train)

# Predict the model

# This gets the predicted probability of having a stroke
train_mlp_preds = clf.predict_proba(X_train)[:, 1]
mlp_preds = clf.predict_proba(X_test)[:, 1]

print("Train Accuracy = ", accuracy(y_train, train_mlp_preds))
print("Train F1 = ", f1(y_train, train_mlp_preds))
print("Train Precision = ", precision(y_train, train_mlp_preds))
print("Train Recall = ", recall(y_train, train_mlp_preds))

print("___________________________")

print("Test Accuracy = ", accuracy(y_test, mlp_preds))
print("Test F1 = ", f1(y_test, mlp_preds))
print("Test Precision = ", precision(y_test, mlp_preds))
print("Test Recall = ", recall(y_test, mlp_preds))

print(clf.get_params())

########################
# Calibration stuff
# n_bins = 30

# # bins = get_bins(n_bins)
# bins = [(0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 1)]

# y_random = []
# for i in range(len(y_test)):
#     if np.random.randint(1000) > 950:
#         y_random.append(1.0)
#     else:
#         y_random.append(0.0)

# # input_data = list(zip(mlp_preds, y_test))
# input_data = list(zip(mlp_preds, y_random))

# binned_data = points_per_bin(data=input_data, bins=bins)
# mean_p, mean_y, counts = per_bin_means(binned_data)
# print(
#     [
#         (float(mp), float(my), int(ct))
#         for mp, my, ct in zip(mean_p, mean_y, counts)
#     ]
# )
# print(f"The ECE is {find_ECE(mean_p, mean_y, counts)}")
# plot_per_bin_means(mean_p, mean_y)
