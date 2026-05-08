import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from stroke_data import get_stroke_data_for_cv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from baseline_models.custom_gridsearch import *
from calibration import *
from stroke_data import get_stroke_data_for_cv


## Setting up the neurel network

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

# I generated this plot with Cursor using the following prompt
# mlp_preds is a list of probabilities between 0 and 1. I want to plot the probabilities on a graph where the x axis is the index of the probabilities sorted and the y axis is the actual probabilities. MAKE SURE NOT TO CHANGE THE ORDER OF THE LIST mlp_preds.  Or if you do, change it back
# Plot sorted probabilities without mutating mlp_preds.
plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), mlp_pairs_sorted[:,0], marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Predicted Probabilities Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

########################
# Calibration stuff

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

y_random = []
for i in range(len(y_test)):
    if np.random.randint(1000) > 951:
        y_random.append(1.0)
    else:
        y_random.append(0.0)

binss = [bins_10, bins_20, best_bins, equal_5]
ys = [mlp_pairs_sorted[:,1], y_random]

for i, bins in enumerate(binss):
    print(f"Bin {i+1} out of {len(binss)}")
    for y_vect in ys:
        input_data = list(zip(mlp_pairs_sorted[:,0], y_vect))

        binned_data = points_per_bin(data=input_data, bins=bins)
        mean_p, mean_y, counts = per_bin_means(binned_data)
        print(
            [
                (float(mp), float(my), int(ct))
                for mp, my, ct in zip(mean_p, mean_y, counts)
            ]
        )
        print(f"The ECE is {find_ECE(mean_p, mean_y, counts)}")
        plot_per_bin_means(mean_p, mean_y)

binned_data = points_per_bin(data=list(zip(mlp_pairs_sorted[:,0], mlp_pairs_sorted[:,1])), bins=best_bins)
mean_p, mean_y, counts = per_bin_means(binned_data)
postprocessed_ps = constant_postprocess(mean_p, mean_y, counts)

plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), postprocessed_ps, marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Predicted Probabilities Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure()
plt.plot(np.arange(len(mlp_pairs_sorted)), mlp_pairs_sorted[:,1], marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Ys Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

print("Origional Accuracy = ", accuracy(mlp_pairs_sorted[:,1], mlp_pairs_sorted[:,0]))
print("Origional F1 = ", f1(mlp_pairs_sorted[:,1], mlp_pairs_sorted[:,0]))
print("Origional Precision = ", precision(mlp_pairs_sorted[:,1], mlp_pairs_sorted[:,0]))
print("Origional Recall = ", recall(mlp_pairs_sorted[:,1], mlp_pairs_sorted[:,0]))

print("___________________________")


print("Postprocessed Accuracy = ", accuracy(mlp_pairs_sorted[:,1], postprocessed_ps))
print("Postprocessed F1 = ", f1(mlp_pairs_sorted[:,1], postprocessed_ps))
print("Postprocessed Precision = ", precision(mlp_pairs_sorted[:,1], postprocessed_ps))
print("Postprocessed Recall = ", recall(mlp_pairs_sorted[:,1], postprocessed_ps))
