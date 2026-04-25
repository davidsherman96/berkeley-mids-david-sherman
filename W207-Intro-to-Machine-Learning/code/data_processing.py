#Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

dataset_source = 'https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset'

scale_columns = ["age", "bmi","HbA1c_level", "blood_glucose_level"]


def retrieve_clean_data(use_dummies=True):
    print(f'Our dataset source: {dataset_source}')

    # Load the diabetes prediction dataset
    df = pd.read_csv('diabetes_prediction_dataset.csv')

    df = df[df["HbA1c_level"] >= 5.7]

    if use_dummies:
        df = pd.get_dummies(df)
    else:
        # TODO: Try dataset using ordinal categories for smoking history and gender.
        pass

    print(f"\nCleaned Dataset:\n{df.head()}")
    return df


def preprocess_data(seed=1234, split=(0.6, 0.2, 0.2), scaler=StandardScaler(), undersample=False, use_dummies=True):
    np.random.seed(seed)
    df = retrieve_clean_data(use_dummies=use_dummies)

    scaler = scaler.fit(df[scale_columns])
    df[scale_columns] = scaler.transform(df[scale_columns])
    df = df.sample(frac=1, random_state=seed)
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]
    print(f"\nX:\n{X.head()}")
    print(f"\ny:\n{y.head()}")

    splits = np.multiply(len(df), split).astype(int)

    def split_df(df):
        training_split = splits[0]
        val_split = splits[0] + splits[1]
        return df.iloc[:training_split], \
               df.iloc[training_split:val_split], \
               df.iloc[val_split:]

    X_train, X_val, X_test = split_df(X)
    y_train, y_val, y_test = split_df(y)

    if undersample:
        neg_indicies = y_train[y_train == 0].index.values
        print(neg_indicies)
        print(y_train[y_train == 1].index.values)
        print(np.random.choice(neg_indicies, len(y_train[y_train == 1]), replace=False))
        random_indices = np.concatenate(
            [np.random.choice(neg_indicies, len(y_train[y_train == 1]), replace=False),
            y_train[y_train == 1].index.values]
        )
        np.random.shuffle(random_indices)
        X_train = X_train.loc[random_indices]
        y_train = y_train.loc[random_indices]
    return X_train, X_val, X_test, y_train, y_val, y_test


def inverse_transform(scaler, data):
    data[scale_columns] = scaler.inverse_transform(data[scale_columns])
    return data
