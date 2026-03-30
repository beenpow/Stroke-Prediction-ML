import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def get_stroke_data():
    # load in imputed stroke data using k=5 and distance weights 
    df = pd.read_csv("data/knn-distance-stroke-data.csv")

    # drop id column
    df = df.drop(["id"], axis=1)

    # split data into features and labels 
    y_df = df['stroke']
    X_df = df.drop(['stroke'], axis=1)

    # transform features and labels into numpy arrays and check shape 
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    # Do an initial 80/20 split
    X_train, X_temp_test, y_train, y_temp_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Do second split to get the val and test splits from the 20
    X_val, X_test, y_val, y_test = train_test_split(X_temp_test, y_temp_test, test_size=0.5, random_state=42, stratify=y_temp_test)

    return X_train, X_val, X_test, y_train, y_val, y_test