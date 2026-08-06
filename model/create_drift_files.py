import pandas as pd


data = pd.read_csv(
    "data/card_transdata.csv"
)


# Create reference dataset (80%)
reference = data.sample(
    frac=0.8,
    random_state=42
)


# Create current dataset (remaining 20%)
current = data.drop(
    reference.index
)


reference.to_csv(
    "data/reference.csv",
    index=False
)


current.to_csv(
    "data/current.csv",
    index=False
)


print("reference.csv created")
print("current.csv created")