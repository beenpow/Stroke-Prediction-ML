import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from src.stroke_data import get_stroke_data_for_cv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# X_train, X_test, y_train, and y_test are defined here.
from src.baseline_models.custom_gridsearch import *
from src.calibration import *


## Setting up the neurel network

calibration_optimized_params = {
            "random_state": 42,
            "alpha": 1.0,
            "solver": "adam",
            "max_iter": 200,
            "hidden_layer_sizes": (100, 24),
            "activation": "relu",
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

# I generated this plot with Cursor using the following prompt
# mlp_preds is a list of probabilities between 0 and 1. I want to plot the probabilities on a graph where the x axis is the index of the probabilities sorted and the y axis is the actual probabilities. MAKE SURE NOT TO CHANGE THE ORDER OF THE LIST mlp_preds.  Or if you do, change it back
# Plot sorted probabilities without mutating mlp_preds.
sorted_mlp_preds = np.sort(mlp_preds.copy())
plt.figure()
plt.plot(np.arange(len(sorted_mlp_preds)), sorted_mlp_preds, marker=".", linestyle="-")
plt.xlabel("Sorted index")
plt.ylabel("Predicted probability")
plt.title("Distribution of Predicted Probabilities Accross Test Samples")
plt.grid(True, alpha=0.3)
plt.show()

########################
# Calibration stuff

bins_50 = get_bins(n_bins=50)
bins_30 = get_bins(n_bins=30)
bins_10 = get_bins(n_bins=10)
bins_custom = [(0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 1)]

y_random = []
for i in range(len(y_test)):
    if np.random.randint(1000) > 951:
        y_random.append(1.0)
    else:
        y_random.append(0.0)

binss = [bins_10, bins_30, bins_50, bins_custom]
ys = [y_test, y_random]

for bins in binss:
    for y_vect in ys:
        input_data = list(zip(mlp_preds, y_vect))

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
