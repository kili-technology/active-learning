import os
import yaml

import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats

from .logger import setup_logger


def set_up_experiment(experiment, folder, logging_lvl=10):
    OUTPUT_DIR = os.path.join(folder, 'results')
    FIGURE_DIR = os.path.join(folder, 'figures')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    logger_name = experiment
    logger = setup_logger(logger_name, OUTPUT_DIR, logging_lvl=logging_lvl)
    return OUTPUT_DIR, FIGURE_DIR, logger, logger_name


def load_config(folder, dataset):
    if dataset == 'mnist':
        config_path = os.path.join(folder, 'mnist.yaml')
    elif dataset == 'cifar':
        config_path = os.path.join(folder, 'cifar.yaml')
    elif dataset == 'pascalvoc_detection':
        config_path = os.path.join(folder, 'pascalvoc_detection.yaml')
    elif dataset == 'pascalvoc_segmentation':
        config_path = os.path.join(folder, 'pascalvoc_segmentation.yaml')
    elif dataset == 'coco_object_detection':
        config_path = os.path.join(folder, 'coco.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def append_to_scores(data, metric, key):
    if type(metric) == dict:
        for subkey, submetric in metric.items():
            return append_to_scores(data, submetric, f'{key}-{subkey}')
    else:
        data[key] = metric
    return data


def extract_df(scores):
    data = []
    for (strategy, run), scores_experiment in scores.items():
        for step_result in scores_experiment:
            experiment_data = {}
            for key, metric in step_result.items():
                experiment_data = append_to_scores(
                    experiment_data, metric, key)
            experiment_data = {
                **experiment_data, 'strategy': strategy, 'run': run}
            data.append(experiment_data)
    df = pd.DataFrame(data)
    return df


def extract_strategy(df, strategy, columns=['val-accuracy', 'run', 'step', 'size_labeled']):
    return df.loc[df.strategy == strategy, columns].groupby('step').mean()


def plot_size_required(df_with_al, df_without_al, plot_dir, points, perf_col='val-accuracy', savename='size_ratio'):
    sizes = df_without_al.loc[:, 'size_labeled']
    acc_to_size_random = interpolate.interp1d(
        df_without_al.loc[:, perf_col], sizes)
    acc_to_size_al = interpolate.interp1d(
        df_with_al.loc[:, perf_col], sizes)
    acc_with_al = acc_to_size_al(points)
    acc_without_al = acc_to_size_random(points)

    _, ax = plt.subplots(num=248, figsize=(12, 5))
    ax.scatter(acc_without_al, acc_with_al)
    alpha, beta, _, _, _ = stats.linregress(acc_without_al, acc_with_al)
    ax.plot(sizes, alpha*sizes+beta, linestyle='-.', color='red')
    ax.plot(sizes, sizes, linestyle='--', color='black', label='baseline')
    ax.grid()
    for i, txt in enumerate(points):
        ax.annotate(f'acc={txt:.0%}', (acc_without_al[i], acc_with_al[i]))
    ax.legend()
    ax.set_ylabel('Number of samples with al')
    ax.set_xlabel('Number of samples without al')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, f'{savename}.png'))
