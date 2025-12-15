import pandas as pd

train = pd.read_csv('data/train.csv')

target_rows = 1_000_000
repeats = -(-target_rows // len(train))  # Ceiling division
train_df = pd.concat([train] * repeats, ignore_index=True).head(target_rows)

train_df.to_csv('data/train-1M.csv')