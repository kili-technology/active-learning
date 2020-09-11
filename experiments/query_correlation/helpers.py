import glob
import os
import itertools

import tqdm
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


NAME_TO_ALGO = {
    'bayesian_bald_sampling': 'Bayesian BALD',
    'bayesian_entropy_sampling': 'Bayesian entropy',
    'coreset': 'K-center greedy',
    'diverse_mini_batch_sampling': 'Diverse Mini Batch',
    'entropy_sampling': 'Entropy',
    'kl_divergence_sampling': 'KL divergence',
    'margin_sampling': 'Margin',
    'random_sampling': 'Random',
    'uncertainty_sampling': 'Uncertainty',
    'al_for_deep_object_detection': 'AL for Deep OD',
    'vote_entropy_sampling': 'Vote entropy'
}

plot_dir = os.path.join(os.path.dirname(__file__), 'figures')


def compute_pairwise_metrics(df):
    ious = []
    correlations = []
    for i in range(len(df)):
        s_i = df.iloc[i]
        for j in range(i+1, len(df)):
            s_j = df.iloc[j]
            run_i = s_i['run']
            run_j = s_j['run']
            df_i = read_file(s_i['file'])
            df_j = read_file(s_j['file'])
            iou, _, _ = get_iou(df_i, df_j)
            ious.append(iou)
            correlation, _ = get_correlation(df_i, df_j)
            correlations.append(correlation)
    metrics = {
        'iou': np.mean(ious),
        'correlation': np.mean(correlations),
        'ioustd': np.std(ious),
        'correlationstd': np.std(correlations)
    }
    return pd.Series(metrics)


def get_iou(df1, df2):
    set1, set2 = set(), set()
    for _, elements in df1.iterrows():
        set1 = set1.union(set(elements))
    for _, elements in df2.iterrows():
        set2 = set2.union(set(elements))
    union = set1.union(set2)
    intersection = set1.intersection(set2)
    return len(intersection) / len(union), len(intersection), len(union)


def get_intersection_steps(df1, df2):
    set1, set2 = set(), set()
    elem2position1, elem2position2 = dict(), dict()
    for i, elements in df1.iterrows():
        set1 = set1.union(set(elements))
        elem2position1 = {**elem2position1, **
                          dict(zip(elements, [i+1]*len(elements)))}
    for i, elements in df2.iterrows():
        set2 = set2.union(set(elements))
        elem2position2 = {**elem2position2, **
                          dict(zip(elements, [i+1]*len(elements)))}
    intersection = set1.intersection(set2)
    step1, step2 = [], []
    for elem in intersection:
        step1.append(elem2position1[elem])
        step2.append(elem2position2[elem])
    return step1, step2


def get_correlation(df1, df2):
    correlation, pval = scipy.stats.pearsonr(*get_intersection_steps(df1, df2))
    return correlation, pval


def read_file(path):
    return pd.read_csv(path, header=None, skiprows=1)


def compare_two_batches(df_batch1, df_batch2):
    ious = []
    correlations = []
    for i in range(len(df_batch1)):
        for j in range(len(df_batch2)):
            df_i = read_file(df_batch1.iloc[i]['file'])
            df_j = read_file(df_batch2.iloc[i]['file'])
            iou, _, _ = get_iou(df_i, df_j)
            ious.append(iou)
            correlation, _ = get_correlation(df_i, df_j)
            correlations.append(correlation)
    metrics = {'iou': np.mean(ious), 'correlation': np.mean(correlations)}
    return pd.Series(metrics)


def read_files(path):
    files = glob.glob(path)
    df_experiments = []
    for file in files:
        file_name = file.split('/')[-1]
        list_file = file_name.split('-')
        algo = list_file[1]
        run = int(list_file[2])
        model = list_file[3]
        df_experiments.append({
            'algorithm': algo,
            'model': model,
            'run': run,
            'file': file
        })
    df_experiments = pd.DataFrame(df_experiments)
    return df_experiments


def translate_algorithms(algos):
    return list(map(lambda x: NAME_TO_ALGO[x], algos))


def produce_figures(path):
    suffix_name = '-'.join(path.split('/')[-3:]).split('.')[0]
    suffix_name = suffix_name.replace('*', '')
    df_experiments = read_files(path)
    results = df_experiments.groupby(
        ['algorithm', 'model']).apply(compute_pairwise_metrics)
    results = results.reset_index()
    results_iou = results.sort_values('iou')
    results_correlation = results.sort_values('correlation')

    plt.figure()
    range_algos = np.arange(len(results.algorithm))
    plt.errorbar(x=range_algos, y=results_iou.iou,
                 yerr=results_iou.ioustd, capsize=10, fmt='.')
    plt.xticks(range_algos, list(
        map(lambda x: NAME_TO_ALGO[x], results_iou.algorithm)), rotation=45)
    plt.ylabel('IOU between trainings')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, f'iou-self-{suffix_name}.png'), dpi=200)

    plt.figure()
    range_algos = np.arange(len(results.algorithm))
    plt.errorbar(x=range_algos, y=results_correlation.correlation,
                 yerr=results_correlation.correlationstd, capsize=10, fmt='.')
    plt.xticks(range_algos, list(
        map(lambda x: NAME_TO_ALGO[x], results_correlation.algorithm)), rotation=45)
    plt.ylabel('Correlation between trainings')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(
        plot_dir, f'correlation-self-{suffix_name}.png'), dpi=200)

    algorithms = list(set(df_experiments.algorithm))
    algorithms = ['random_sampling'] + \
        [x for x in algorithms if x != 'random_sampling']
    algo2i = dict(zip(algorithms, range(len(algorithms))))
    data = np.zeros((len(algo2i), len(algo2i), 2))
    for algo1, algo2 in tqdm.tqdm(itertools.combinations(algorithms, 2)):
        df_batch1 = df_experiments.loc[df_experiments.algorithm == algo1]
        df_batch2 = df_experiments.loc[df_experiments.algorithm == algo2]
        results = compare_two_batches(df_batch1, df_batch2)
        iou, correlation = results['iou'], results['correlation']
        data[algo2i[algo1], algo2i[algo2], 0] = iou
        data[algo2i[algo1], algo2i[algo2], 1] = correlation
        data[algo2i[algo2], algo2i[algo1], 0] = iou
        data[algo2i[algo2], algo2i[algo1], 1] = correlation

    df_corr = pd.DataFrame(data[:, :, 0], index=translate_algorithms(
        algorithms), columns=translate_algorithms(algorithms))
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df_corr, mask=mask, cmap="viridis_r", vmax=df_corr.max().max(), vmin=0,
                square=True, linewidths=.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'iou-{suffix_name}.png'), dpi=200)

    df_corr = pd.DataFrame(data[:, :, 1], index=translate_algorithms(
        algorithms), columns=translate_algorithms(algorithms))
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df_corr, mask=mask, cmap="vlag", vmax=df_corr.max().max(), center=0, vmin=df_corr.min().min(),
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir, f'correlation-{suffix_name}.png'), dpi=200)
