"""
This experiment tests the best size of the training data
"""
import os
import logging
import pickle

import numpy as np

import al
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment, load_config 
from al.experiments import set_up_learner



EXPERIMENT_NAME = 'initial_training_data'
FOLDER_PATH = os.path.expanduser(f'~/Documents/active-learning/experiments/{EXPERIMENT_NAME}')
DATASETS = ['mnist', 'cifar', 'pascalvoc_detection']
dataset_to_initsizes = {
    'mnist': [10, 30, 100, 300, 1000],
    'cifar': [100, 300, 1000, 3000, 10000],
    'pascalvoc_detection': [100, 300, 500, 1000]
}
REPEATS = 2


OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(EXPERIMENT_NAME)

def run_single_experiment(dataset_name, init_size):
    logger.info(f'INITIAL SIZE : {init_size}')
    config = load_config(FOLDER_PATH, dataset_name)
    setupper = set_up_learner(dataset_name)
    config['active_learning']['output_dir'] = OUTPUT_DIR
    config['active_learning']['init_size'] = init_size
    config['experiment']['logger_name'] = logger_name
    logger.debug('Getting dataset and learner')
    dataset, learner = setupper(config, OUTPUT_DIR, logger)
    logger.debug('Getting trainer')
    trainer = ActiveTrain(learner, dataset, config['experiment']['strategy'], logger_name)
    logger.debug('Training...')
    scores = None
    # scores = trainer.train(config['train_parameters'], **config['active_learning'])
    logger.debug('Done training...')
    logger.info('-------------------------')
    return scores

def run_on_dataset(dataset):
    logger.info('-------------------------')
    logger.info('--LAUNCHING EXPERIMENT--')
    logger.info(f'--DATASET {dataset}--')
    logger.info('-------------------------')
    data_scores = []
    for init_size in dataset_to_initsizes[dataset]:
        for repeat in range(REPEATS):
            logger.info('-------------------------')
            logger.info(f'--REPEAT #{repeat+1}/{REPEATS}--')
            logger.info('-------------------------')
            scores = run_single_experiment(dataset_name, init_size)
            data_scores.append({
                'repeat': repeat,
                'score': scores,
                'init_size': init_size
            })
    return data_scores

logger.info('-------------------------')
logger.info('--LAUNCHING EXPERIMENTS--')
logger.info('-------------------------')

for dataset_name in DATASETS:
    score_data = run_on_dataset(dataset_name)
    with open(f'{OUTPUT_DIR}/scores_{dataset_name}.pickle', 'wb') as f:
        pickle.dump(score_data, f)

