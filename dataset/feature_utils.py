# dataset/feature_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_dataset(df, feature_cols, target_cols):
    """
    Encodes 'predicate' and 'semantic' as integers, extracts feature matrix and targets.

    Returns:
        X: Feature matrix (N x D)
        Y: Target values (N x 1)
        df: Encoded DataFrame
        feature_cols: List of used feature columns
    """
    label_enc_pred = LabelEncoder()
    df["encoded_predicate"] = label_enc_pred.fit_transform(df["predicate"])

    label_enc_sem = LabelEncoder()
    df["encoded_semantic"] = label_enc_sem.fit_transform(df["semantic"])

    X = df[feature_cols].values
    Y = df[target_cols].values

    return X, Y, df, feature_cols


def split_dataset(df):
    """
    Splits dataset into Train (3 parts), Validation (2 parts), and Test (2 parts).
    Ratio: 3:2:2
    """
    train_ratio = 3 / 7
    val_ratio = 2 / 7
    test_ratio = 2 / 7

    df_train, df_temp = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    return df_train, df_val, df_test
