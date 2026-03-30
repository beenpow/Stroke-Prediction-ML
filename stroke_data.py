import pandas as pd
from sklearn.model_selection import train_test_split

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
    # print("Shape of encoded data with stroke col dropped: ", X.shape)
    # print("Shape of encoded data with stroke col dropped: ", y.shape)

    # Do an initial 80/20 split
    X_train, X_temp_test, y_train, y_temp_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # print("X_train shape = ", X_train.shape)
    # print("X_temp_test shape = ", X_temp_test.shape)
    # print("y_train shape = ", y_train.shape)
    # print("y_temp_test shape = ", y_temp_test.shape)

    # print("Checking sum of elems = ", X_train.shape[0] + X_temp_test.shape[0])

    half_idx = int(X_temp_test.shape[0] / 2)

    # get the val and test splits by splitting the temp_test numpy arrays at the half_idx
    X_val = X_temp_test[:half_idx]
    X_test = X_temp_test[half_idx:]

    y_val = y_temp_test[:half_idx]
    y_test = y_temp_test[half_idx:]

    return X_train, X_val, X_test, y_train, y_val, y_test