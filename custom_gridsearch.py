from itertools import product

from sklearn.neural_network import MLPClassifier

from stroke_data import get_stroke_data

X_train, X_val, X_test, y_train, y_val, y_test = get_stroke_data("data/knn-standardize-distance.csv")

def get_config_arr(params):
    if not params:
        return []
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]

def harmonic_mean(a,b):
    return 2/(1/a+1/b)

def accuracy(y_arr, p_arr):
    n = len(y_test)

    tn = sum((1-y_arr[i]) * (1-p_arr[i]) for i in range(n))
    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    return (tn+tp)/(fn+fp+tn+tp)

def precision(y_arr, p_arr):
    n = len(y_test)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    return tp/(tp+fp)

def recall(y_arr, p_arr):
    n = len(y_test)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))

    return tp/(tp+fn)

def f1(y_arr, p_arr):
    n = len(y_test)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return harmonic_mean(precision, recall)

def mlp_fake_grid_search(params, scorer):
    config_arr = get_config_arr(params)
    best_score = -1
    best_configs = {}
    for configuration in config_arr:
        model = MLPClassifier(random_state=1, **configuration)
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        score = scorer(y_val, val_probs)
        if (score > best_score):
            best_configs = configuration
            best_score = score
    return best_configs, best_score

