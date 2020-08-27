import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from al.helpers.experiment import set_up_experiment, load_config


FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, _, _ = set_up_experiment(
    __file__, FOLDER_PATH, logging_lvl=20)
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def get_results_from(dataset, model_name):
    with open(f'{OUTPUT_DIR}/scores-{dataset}-{model_name}.pickle', 'rb') as f:
        raw_scores = pickle.load(f)
        data = []
        for (query_size, experiment_number), scores_experiment in raw_scores.items():
            print(query_size)
            for step_result in scores_experiment:
                val_step_result = step_result['val']
                step = step_result['step']
                size_labeled = step_result['size_labeled']
                step_time = step_result['times']['active_iter']
                data.append(
                    {'query_size': query_size,
                     'experiment': experiment_number,
                     'step': step,
                     'size_labeled': size_labeled,
                     'step_time': step_time,
                     ** val_step_result})
        df = pd.DataFrame(data)
        df.query_size = df.query_size.astype(str)
        return df


def plot_accuracy_versus_labeled(df):
    plt.figure(num=0, figsize=(12, 5))
    sns.lineplot(x='size_labeled', y='accuracy', hue='query_size', data=df)
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig(os.path.join(PLOT_DIR, 'accuracy.png'))


if __name__ == '__main__':
    df = get_results_from('mnist', 'simple_cnn')
    print(df)
    print(df.query_size.value_counts())
    print(df.groupby('query_size').accuracy.mean())
    plot_accuracy_versus_labeled(df)
