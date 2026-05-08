# SMOTE train only, test untouched. needs: pip install imbalanced-learn
# run: python src/smote/smote_baselines.py

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
REPO_ROOT = _here.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pprint import pprint

import numpy as np
from imblearn.over_sampling import SMOTE

from stroke_data import get_stroke_data_for_cv
from all_baselines import run_all_baselines


def main():
    csv_rel = "data/knn-standardize-distance.csv"
    X_train, X_test, y_train, y_test = get_stroke_data_for_cv(csv_rel)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    uniq, cnt = np.unique(y_train, return_counts=True)
    print(
        f"Train shape before SMOTE: {X_train.shape}, after: {X_train_res.shape}"
    )
    print(f"Class counts on train (before SMOTE): {dict(zip(uniq.tolist(), cnt.tolist()))}")

    print("\nSMOTE train, default baselines (no class_weight on lr/svm/rf):\n")
    pprint(
        run_all_baselines(
            X_train_res, X_test, y_train_res, y_test
        )
    )


if __name__ == "__main__":
    main()
