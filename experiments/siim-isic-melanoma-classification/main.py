"""
From https://www.kaggle.com/abhishek/melanoma-detection-with-pytorch
"""

import os
import yaml
import pickle

import tqdm
import numpy as np
import pandas as pd


from data_processing import MelanomaDataset, get_train_val_datasets
from model import SEResnext50_32x4dLearner

from al.train.active_train import ActiveTrain
from al.helpers.experiment import load_config, set_up_experiment


with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(
    __file__, FOLDER_PATH, logging_lvl=20)


train_ds, val_ds = get_train_val_datasets()


def set_up_dataset_learner():
    dataset = MelanomaDataset(
        train_ds, val_ds, n_init=config['active_learning']['init_size'])
    print('Initial labeled size :', len(dataset.get_labeled()))
    print('Unlabeled size :', len(dataset.get_unlabeled()))
    print('Validation size :', len(dataset.get_validation_dataset()))
    validation_targets = dataset.get_validation_dataset().targets
    print('Targets validation :', pd.value_counts(validation_targets))
    learner = SEResnext50_32x4dLearner(device=0, logger_name=logger_name)
    return dataset, learner


score_data = {}

for i in range(config['experiment']['repeats']):
    logger.info('---------------------------')
    logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
    logger.info('---------------------------')
    for strategy in config['experiment']['strategies']:
        dataset, learner = set_up_dataset_learner()
        logger.info('---------------------------')
        logger.info(f'----STRATEGY : {strategy}----')
        logger.info('---------------------------')
        trainer = ActiveTrain(learner, dataset, strategy, logger_name=__file__)
        scores = trainer.train(
            config['train_parameters'], **config['active_learning'])
        score_data[(strategy, i)] = scores
        logger.info(f'----DONE----\n')
    logger.info('---------------------------')
    logger.info(f'--------DONE--------')
    logger.info('---------------------------\n\n\n')

# dataset, learner = set_up_dataset_learner()

# logger.info('WITH ACTIVE LEARNING')
# trainer = ActiveTrain(
#     learner, dataset, method='uncertainty_sampling', logger_name=__file__)
# scores_with_al = trainer.train(
#     config['train_parameters'], **config['active_learning'])

# dataset, learner = set_up_dataset_learner()

# logger.info('WITHOUT ACTIVE LEARNING')
# trainer = ActiveTrain(
#     learner, dataset, method='random_sampling', logger_name=__file__)
# scores_without_al = trainer.train(
#     config['train_parameters'], **config['active_learning'])

# score_data = {'with_al': scores_with_al, 'without_al': scores_without_al}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
if config['experiment']['save_results']:
    with open(f'{OUTPUT_DIR}/scores.pickle', 'wb') as f:
        pickle.dump(score_data, f)
