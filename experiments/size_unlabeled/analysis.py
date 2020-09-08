import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


EXPERIMENT_NAME = os.path.dirname(__file__)
model_name = 'simple_cnn'
OUTPUT_DIR = f'{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'{EXPERIMENT_NAME}/figures'
plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'rb') as f:
    scores = pickle.load(f)

data = []
for (strategy, experiment_number, size), scores_experiment in scores.items():
    for step_result in scores_experiment:
        val_step_result = step_result['val']
        step = step_result['step']
        data.append(
            {'strategy': strategy,
             'experiment': experiment_number,
             'step': step,
             'size': size,
             **val_step_result})

df = pd.DataFrame(data)

print(df)
max_step = df.step.max()
df_final = df.loc[df.step == max_step]

plt.figure(num=0, figsize=(8, 5))
sns.boxplot(x='size', y='accuracy', hue='strategy', data=df_final, width=0.6)
plt.ylabel('Accuracy')
plt.show()
plt.savefig(os.path.join(plot_dir, 'box_accuracy.png'), dpi=300)
