import sys
from pathlib import Path

# cwd can be repo root, src/, or src/feature_selection/ — walk up until src/stroke_data.py exists
_here = Path().resolve()
REPO_ROOT = _here
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "src" / "stroke_data.py").is_file():
        break
    REPO_ROOT = REPO_ROOT.parent
else:
    raise FileNotFoundError("Could not find src/stroke_data.py (open this project from the repo folder).")

sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def logistic_regression(X_train, X_test, y_train, y_test):
    results = {}

    lr = LogisticRegression(C=0.001, solver='liblinear')

    lr.fit(X=X_train, y=y_train)

    lr_train_preds = lr.predict(X_train)
    lr_preds = lr.predict(X_test)

    results = {
        "train": {"accuracy": accuracy_score(y_train, lr_train_preds),
                    "f1": f1_score(y_train, lr_train_preds),
                    "precision": precision_score(y_train, lr_train_preds),
                    "recall": recall_score(y_train, lr_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, lr_preds),
                    "f1": f1_score(y_test, lr_preds),
                    "precision": precision_score(y_test, lr_preds),
                    "recall": recall_score(y_test, lr_preds)},
    }

    return results

def support_vector_machine(X_train, X_test, y_train, y_test):
    svm = SVC(C=10.0, gamma=1, kernel='rbf')

    svm.fit(X=X_train, y=y_train)

    svm_train_preds = svm.predict(X_train)
    svm_preds = svm.predict(X_test)

    results = {
        "train": {"accuracy": accuracy_score(y_train, svm_train_preds),
                    "f1": f1_score(y_train, svm_train_preds),
                    "precision": precision_score(y_train, svm_train_preds),
                    "recall": recall_score(y_train, svm_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, svm_preds),
                    "f1": f1_score(y_test, svm_preds),
                    "precision": precision_score(y_test, svm_preds),
                    "recall": recall_score(y_test, svm_preds)},
    }

    return results

def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(bootstrap=False, max_depth=None, max_features=None, max_leaf_nodes=None, n_estimators=15)

    rf.fit(X=X_train, y=y_train)

    rf_train_preds = rf.predict(X_train)
    rf_preds = rf.predict(X_test)

    results = {
        "train": {"accuracy": accuracy_score(y_train, rf_train_preds),
                    "f1": f1_score(y_train, rf_train_preds),
                    "precision": precision_score(y_train, rf_train_preds),
                    "recall": recall_score(y_train, rf_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, rf_preds),
                    "f1": f1_score(y_test, rf_preds),
                    "precision": precision_score(y_test, rf_preds),
                    "recall": recall_score(y_test, rf_preds)},
    }

    return results

def xg_boost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(eta=1, gamma=2, reg_lambda=0.5, max_depth=6, objective='binary:logistic', subsample=0.1)

    xgb.fit(X=X_train, y=y_train)

    xgb_train_preds = xgb.predict(X_train)
    xgb_preds = xgb.predict(X_test)

    results = {
        "train": {"accuracy": accuracy_score(y_train, xgb_train_preds),
                    "f1": f1_score(y_train, xgb_train_preds),
                    "precision": precision_score(y_train, xgb_train_preds),
                    "recall": recall_score(y_train, xgb_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, xgb_preds),
                    "f1": f1_score(y_test, xgb_preds),
                    "precision": precision_score(y_test, xgb_preds),
                    "recall": recall_score(y_test, xgb_preds)},
    }

    return results

def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB(priors=None, var_smoothing=1e-11)

    gnb.fit(X=X_train, y=y_train)

    gnb_preds = gnb.predict(X_test)
    gnb_train_preds = gnb.predict(X_train)

    results = {
        "train": {"accuracy": accuracy_score(y_train, gnb_train_preds),
                    "f1": f1_score(y_train, gnb_train_preds),
                    "precision": precision_score(y_train, gnb_train_preds),
                    "recall": recall_score(y_train, gnb_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, gnb_preds),
                    "f1": f1_score(y_test, gnb_preds),
                    "precision": precision_score(y_test, gnb_preds),
                    "recall": recall_score(y_test, gnb_preds)},
    }

    return results

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs= None, n_neighbors=1, p=2) 
    
    knn.fit(X=X_train, y=y_train)

    knn_preds = knn.predict(X_test)
    knn_train_preds = knn.predict(X_train)


    results = {
        "train": {"accuracy": accuracy_score(y_train, knn_train_preds),
                    "f1": f1_score(y_train, knn_train_preds),
                    "precision": precision_score(y_train, knn_train_preds),
                    "recall": recall_score(y_train, knn_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, knn_preds),
                    "f1": f1_score(y_test, knn_preds),
                    "precision": precision_score(y_test, knn_preds),
                    "recall": recall_score(y_test, knn_preds)},
    }

    return results

def multi_layer_perceptron(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=200, solver='adam')

    mlp.fit(X=X_train, y=y_train)

    mlp_preds = mlp.predict(X_test)
    mlp_train_preds = mlp.predict(X_train)

    results = {
        "train": {"accuracy": accuracy_score(y_train, mlp_train_preds),
                    "f1": f1_score(y_train, mlp_train_preds),
                    "precision": precision_score(y_train, mlp_train_preds),
                    "recall": recall_score(y_train, mlp_train_preds)},
        "test": {"accuracy": accuracy_score(y_test, mlp_preds),
                    "f1": f1_score(y_test, mlp_preds),
                    "precision": precision_score(y_test, mlp_preds),
                    "recall": recall_score(y_test, mlp_preds)},
    }

    return results


def run_all_baselines(X_train, X_test, y_train, y_test):
    results = {}

    results['lr'] = logistic_regression(X_train, X_test, y_train, y_test)
    results['svm'] = support_vector_machine(X_train, X_test, y_train, y_test)
    results['rf'] = random_forest(X_train, X_test, y_train, y_test)
    results['xgb'] = xg_boost(X_train, X_test, y_train, y_test)
    results['nb'] = naive_bayes(X_train, X_test, y_train, y_test)
    results['knn'] = k_nearest_neighbors(X_train, X_test, y_train, y_test)
    results['mlp'] = multi_layer_perceptron(X_train, X_test, y_train, y_test)

    return results