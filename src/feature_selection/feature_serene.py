import sklearn
import scipy
import numpy as np
from stroke_data import get_stroke_data_for_cv, get_stroke_data

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

import sklearn
import scipy
import numpy as np
from stroke_data import get_stroke_data_for_cv, get_stroke_data

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")

def logistic_regression(X_train, X_test, y_train, y_test):
    results = {}

    lr = LogisticRegression(random_state=42, C=0.1, class_weight='balanced', solver='newton-cg')

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
    svm = SVC(C=10.0, class_weight='balanced', gamma=0.01, kernel='rbf')

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
    rf = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', max_depth=15, max_features=None, max_leaf_nodes=15, n_estimators=20)

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

def xg_boost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(eta=1, gamma=1, reg_lambda=0.5, max_depth=15, objective='binary:logistic', subsample=1)

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
    gnb = GaussianNB(priors=None, var_smoothing=1e-09)

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



def multi_layer_perceptron(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(random_state=42, alpha=0.001, hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=200, solver='adam')

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




def feature_selections(features):
    X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")
    features = np.abs(features)
    mean = np.mean(features)
    idx = np.where(features < features.mean())[0]
    X_train_reduced = np.delete(X_train, idx, axis=1)

    return X_train, y_train


def logistic_regression_features():
    X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")
    X_train.shape
    lr = LogisticRegression(random_state=42, C=0.1, class_weight='balanced', solver='newton-cg')
    lr = lr.fit(X_train, y_train)
    lr.coef_
    model = SelectFromModel(lr, prefit=True)
    X_new = model.transform(X_train)
    X_new.shape
    features = lr.coef_
    X_train, y_train = feature_selections(features)
  
    return logistic_regression(X_train, X_test, y_train, y_test)


def support_vector_machine_features():
   
    X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")
    svm = SVC(C=10.0, class_weight='balanced', gamma=0.01, kernel='linear')
    svm.fit(X_train, y_train)

    model = SelectFromModel(svm, prefit=True)
    X_new = model.transform(X_train)
    features =  svm.coef_
    X_train, y_train = feature_selections(features)

    return support_vector_machine(X_train, X_test, y_train, y_test)

def random_forest_features():

    X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")
    X_train.shape
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_train)
    X_new.shape
    features = clf.feature_importances_
    X_train, y_train = feature_selections(features)
    print(X_train.shape)
    return random_forest(X_train, X_test, y_train, y_test)


def xg_boost_features():

    X_train, X_test, y_train, y_test = get_stroke_data_for_cv("../data/knn-standardize-distance.csv")
    xgb = XGBClassifier(eta=1, gamma=1, reg_lambda=0.5, max_depth=15, objective='binary:logistic', booster='gblinear', subsample=0.5)
    xgb.fit(X_train, y_train)

    model = SelectFromModel(xgb, prefit=True)
    X_new = model.transform(X_train)
    features = xgb.coef_
    X_train, y_train = feature_selections(features)
     
    return xg_boost(X_train, X_test, y_train, y_test)



def run_all_baselines_features():
    results = {}

    results['lr'] = logistic_regression_features()
    results['svm'] = support_vector_machine_features()
    results['rf'] = random_forest_features()
    results['xgb'] = xg_boost_features()
    #results['nb'] = naive_bayes(X_train, X_test, y_train, y_test)
    #results['knn'] = k_nearest_neighbors(X_train, X_test, y_train, y_test)
    #results['mlp'] = multi_layer_perceptron(X_train, X_test, y_train, y_test)
    print(results)
    return results

run_all_baselines_features()