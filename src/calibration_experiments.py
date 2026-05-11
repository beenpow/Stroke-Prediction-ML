###### Imports ######

import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from stroke_data import get_stroke_data_for_cv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from f1_helper import *
from calibration import *
from stroke_data import get_stroke_data_for_cv


###### Setting up the neurel network ######

X_train, X_test, y_train, y_test = get_stroke_data_for_cv("data/knn-standardize-distance.csv")

calibration_optimized_params = {
            "random_state": 42,
            "alpha": 1.0,
            "solver": "adam",
            "max_iter": 200,
            "hidden_layer_sizes": (100, 100, 100, 100, 100)
}

hyperparameter_optimized_params = {
            "random_state": 42,
            "alpha": 0.001,
            "solver": "adam",
            "max_iter": 200,
            "hidden_layer_sizes": (100, 100, 100, 100, 100),
}


mlp = MLPClassifier(**calibration_optimized_params)

mlp.fit(X_train, y_train)
mlp_preds = mlp.predict_proba(X_test)[:, 1]
# I sorted mlp_preds with the following prompt (Notice I said a false statement in this prompt):
# mlp_preds is a list of (p,y). I want to sort this list by p's but keep each p with its coorisponding y.
mlp_pairs = list(zip(mlp_preds, y_test))
mlp_pairs_sorted = np.asarray(sorted(mlp_pairs, key=lambda t: t[0]))

prediction_arr = mlp_pairs_sorted[:,0]
y_arr = mlp_pairs_sorted[:,1]



###### Plotting the distributions of the probability predictions ######

# I generated this plot with Cursor using the following prompt
# mlp_preds is a list of probabilities between 0 and 1. I want to plot the probabilities on a graph where the x axis is the index of the probabilities sorted and the y axis is the actual probabilities. MAKE SURE NOT TO CHANGE THE ORDER OF THE LIST mlp_preds.  Or if you do, change it back
# Plot sorted probabilities without mutating mlp_preds.
plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), prediction_arr, marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Predicted Probabilities Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()



###### Calibration stuff ######

# Setting up bins
bins_30 = get_bins(n_bins=30)
bins_20 = get_bins(n_bins=20)
bins_10 = get_bins(n_bins=10)
equal_4 = custom_equal_size_bins(mlp_pairs_sorted, 4)
equal_5 = custom_equal_size_bins(mlp_pairs_sorted, 5)
equal_6 = custom_equal_size_bins(mlp_pairs_sorted, 6)
equal_7 = custom_equal_size_bins(mlp_pairs_sorted, 7)
equal_8 = custom_equal_size_bins(mlp_pairs_sorted, 8)
equal_9 = custom_equal_size_bins(mlp_pairs_sorted, 9)
equal_10 = custom_equal_size_bins(mlp_pairs_sorted, 10)
best_bins = finest_equal_size_monotonic_bins(mlp_pairs_sorted)

# As a control, we tested our predictors against a raondom array which has the same fraction of positive labels.
y_random = []
for i in range(len(y_test)):
    if np.random.randint(1000) > 951:
        y_random.append(1.0)
    else:
        y_random.append(0.0)

# Testing calibration and plotting a calibration curve for each y array and each binning scheme.
binss = [bins_10, bins_20, best_bins, equal_5]
ys = [y_arr, y_random]

for i, bins in enumerate(binss):
    print(f"Bin {i+1} out of {len(binss)}")
    for y_vect in ys:
        input_data = list(zip(mlp_pairs_sorted[:,0], y_vect))

        binned_data = points_per_bin(data=input_data, bins=bins)
        mean_p, mean_y, counts = per_bin_means(binned_data)
        # print(
        #     [
        #         (float(mp), float(my), int(ct))
        #         for mp, my, ct in zip(mean_p, mean_y, counts)
        #     ]
        # )
        print(f"The ECE is {find_ECE(mean_p, mean_y, counts)}")
        plot_per_bin_means(mean_p, mean_y)



###### Calibration Postprocessing ######

binned_data = points_per_bin(data=list(mlp_pairs_sorted), bins=best_bins)
mean_p, mean_y, counts = per_bin_means(binned_data)
postprocessed_prediction_arr = constant_postprocess(mean_p, mean_y, counts)

# Plotting the distribution of postprocessed predictors. This should look like horizontal lines when using constant_postprocess
plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), postprocessed_prediction_arr, marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Predicted Probabilities Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

# Plotting the distribution of positive labels.
plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), y_arr, marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Ys Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

# Printing results of calibration postprocessing
print("Original Accuracy = ", accuracy(y_arr, prediction_arr))
print("Original F1 = ", f1(y_arr, prediction_arr))
print("Original Precision = ", precision(y_arr, prediction_arr))
print("Original Recall = ", recall(y_arr, prediction_arr))
print("___________________________")
print("Postprocessed Accuracy = ", accuracy(y_arr, postprocessed_prediction_arr))
print("Postprocessed F1 = ", f1(y_arr, postprocessed_prediction_arr))
print("Postprocessed Precision = ", precision(y_arr, postprocessed_prediction_arr))
print("Postprocessed Recall = ", recall(y_arr, postprocessed_prediction_arr))
