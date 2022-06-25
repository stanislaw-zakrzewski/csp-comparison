import pandas as pd

df = pd.read_csv('../individual_classes.csv')
print(df.groupby(['structure'])['accuracy'].mean())
