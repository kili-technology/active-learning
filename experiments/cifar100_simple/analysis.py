import os
import pickle
import yaml

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from al.helpers.experiment import *
from al.experiments import set_up_learner
from al.helpers.logger import setup_logger


EXPERIMENT_NAME = os.path.dirname(__file__)
model_name = 'mobilenet'
OUTPUT_DIR = f'{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'{EXPERIMENT_NAME}/figures'
plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

analyze_results = False
analyze_queries = False
analyze_sizes = True

if analyze_queries:
    dataset = 'cifar'
    config = load_config(EXPERIMENT_NAME, dataset)
    logger = setup_logger('analysis', OUTPUT_DIR, logging_lvl=20)
    config['experiment']['logger_name'] = 'analysis'
    dataset, learner = set_up_learner(dataset)(config, OUTPUT_DIR, logger)
    labels = np.array([x[1] for x in dataset.dataset])
    train_distribution = pd.value_counts(labels).sort_values()
    print('train_distribution', train_distribution)
    pbar = tqdm.tqdm(total=config['experiment']['repeats']*len(
        config['experiment']['strategies'])*config['active_learning']['n_iter'])
    list_data = []
    for i in range(config['experiment']['repeats']):
        for strategy in config['experiment']['strategies']:
            queries_name = f'queries-{strategy}-{i}-{model_name}.txt'
            queries = pd.read_csv(f'{OUTPUT_DIR}/{queries_name}')
            for j, query in queries.iterrows():
                pbar.update(1)
                new_query = query.dropna().astype(int)
                query_labels = labels[list(new_query)]
                label_count = dict(pd.value_counts(query_labels).astype(int))
                label_count = {str(k): label_count.get(k, 0)
                               for k in range(100)}
                data = {'repeat': i, 'strategy': strategy,
                        'query': j, **label_count}
                list_data.append(data)
    pbar.close()
    df = pd.DataFrame(list_data).sort_index(axis=1)
    correlation_with_query = np.zeros(
        (100, len(config['experiment']['strategies'])))
    for i in range(100):
        df_i = df.loc[:, [str(i), 'query', 'repeat', 'strategy']]
        df_i = df_i.groupby(['strategy', 'repeat']).corr()
        df_i = df_i.loc[(slice(None), slice(None), str(i)), 'query']
        df_i = df_i.reset_index().groupby('strategy').mean()
        for j, strategy in enumerate(config['experiment']['strategies']):
            correlation_with_query[i, j] = df_i.loc[strategy, 'query']

    df_pair = pd.DataFrame(
        correlation_with_query, columns=config['experiment']['strategies'], index=np.arange(100))
    print(df_pair)

    plt.figure(num=0, figsize=(12, 5))
    for j, strategy in enumerate(config['experiment']['strategies']):
        sns.distplot(correlation_with_query[:, j], label=strategy, hist=False)
    plt.title(
        'Distribution of correlation between number \n of elements of class queried and query step')
    plt.xlabel('Id of the class')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'correlation_queries.png'))

    plt.figure(num=1, figsize=(12, 8))
    g = sns.pairplot(data=df_pair)
    for row_ax in g.axes:
        for ax in row_ax:
            ax.set_xlim((-1, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'pairplot.png'))

if analyze_results:

    with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'rb') as f:
        scores = pickle.load(f)

    data = []
    for (strategy, experiment_number), scores_experiment in scores.items():
        for step_result in scores_experiment:
            val_step_result = step_result['val']
            step = step_result['step']
            data.append(
                {'strategy': strategy,
                 'experiment': experiment_number,
                 'step': step,
                 **val_step_result})

    df = pd.DataFrame(data)

    print(df)

    df = df.loc[~df.strategy.isin(
        ['coreset', 'margin_sampling', 'bayesian_bald_sampling'])]

    plt.figure(num=0, figsize=(12, 5))
    sns.lineplot(x='step', y='accuracy', hue='strategy', data=df)
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'accuracy.png'))

    plt.figure(num=1, figsize=(12, 5))
    sns.lineplot(x='step', y='loss', hue='strategy', data=df)
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'loss.png'))

if analyze_sizes:
    with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'rb') as f:
        scores = pickle.load(f)
    df = extract_df(scores)

    df_random = extract_strategy(df, 'random_sampling')
    df_al = extract_strategy(df, 'entropy_sampling')

    plot_size_required(df_al, df_random, plot_dir, points=[
        0.2, 0.3, 0.4, 0.45])
