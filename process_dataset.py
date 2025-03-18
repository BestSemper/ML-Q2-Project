import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

files = glob('data/*')

for f in files:
    if 'train' in f or 'test' in f:
        continue
    dataset = pd.read_csv(f)
    f = f.split('.')[0]
    # Replace missing values with mode
    for column in dataset.columns[:-1]:
        mode_value = dataset[column].mode()[0]
        dataset[column] = dataset[column].replace('?', mode_value)
    # Remove instances with missing class labels
    dataset = dataset[dataset['class'] != '?']
    train_set, test_set = train_test_split(dataset, test_size=0.2, stratify=dataset['class'], random_state=0)
    train_set.to_csv(f"{f}_train.csv", index=False)
    test_set.to_csv(f"{f}_test.csv", index=False)