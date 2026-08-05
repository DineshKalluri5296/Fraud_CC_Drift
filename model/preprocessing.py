import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset
    """
    return pd.read_csv(filepath)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows
    """
    return df.drop_duplicates()


def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values
    """
    return df.dropna()


def preprocess_data(filepath: str):
    """
    Complete preprocessing pipeline
    """

    df = load_data(filepath)

    df = remove_duplicates(df)

    df = remove_missing(df)

    X = df.drop("fraud", axis=1)

    y = df["fraud"]

    return X, y