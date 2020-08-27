import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



EXPERIMENT_NAME = 'pascal_voc_object_detection'
model_name = 'mobilenet_v2'
# model_name = 'simple_cnn'
OUTPUT_DIR = f'experiments/{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'experiments/{EXPERIMENT_NAME}/figures'

with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'rb') as f:
    scores = pickle.load(f)

# print(scores)

data = []
for (strategy, experiment_number), scores_experiment in scores.items():
    for step_result in scores_experiment:
        val_step_result = step_result['val']
        step = step_result['step']
        data.append(
            {'strategy': strategy,
            'experiment': experiment_number,
            'step': step,
            **val_step_result,
            **val_step_result['metrics']})

df = pd.DataFrame(data)

print(df)

plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

plt.figure(num=0, figsize=(12, 5))
sns.lineplot(x='step', y='mAP', hue='strategy', data=df)
plt.ylabel('Accuracy')
plt.show()
plt.savefig(os.path.join(plot_dir, 'accuracy.png'))