import pandas as pd

df = pd.read_csv('all_scores_channels.csv')
print(df.groupby(['channels'])['accuracy'].mean())
