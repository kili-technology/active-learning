import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from al.helpers.experiment import *


EXPERIMENT_NAME = os.path.dirname(__file__)
OUTPUT_DIR = f'{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'{EXPERIMENT_NAME}/figures'
plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

with open(f'{OUTPUT_DIR}/scores.pickle', 'rb') as f:
    scores = pickle.load(f)


df = extract_df(scores)


def extract_ratio_0to1(s):
    index = s.index.values
    index = [int(i) for i in index]
    values = s.values
    i2n = dict(zip(index, values))
    return i2n[0] / i2n[1]


df['ratio'] = df.loc[:, 'train_metrics-target_distribution'].apply(
    extract_ratio_0to1)


# plt.figure(num=0, figsize=(12, 5))
# sns.lineplot(x='size_labeled', y='val-auc',
#              hue='strategy', data=df, markers=True, style="strategy", dashes=False)
# plt.ylabel('Accuracy')
# plt.show()
# plt.savefig(os.path.join(plot_dir, 'accuracy.png'))


plt.figure(num=0, figsize=(12, 5))
sns.lineplot(x='size_labeled', y='ratio',
             hue='strategy', data=df, markers=True, style="strategy", dashes=False)
plt.title('Ratio of the number of negative examples over positive ones')
plt.show()
plt.savefig(os.path.join(plot_dir, 'ratio.png'))


columns = ['val-auc', 'run', 'step', 'size_labeled']
df_random = extract_strategy(df, 'random_sampling', columns)
df_al = extract_strategy(df, 'uncertainty_sampling', columns)

print(df_random.describe())


plot_size_required(df_al, df_random, plot_dir, points=[
    0.67, 0.72, 0.78], perf_col='val-auc')
