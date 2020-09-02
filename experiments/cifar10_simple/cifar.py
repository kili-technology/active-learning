import os
import logging
import pickle

import numpy as np

import al
from al.dataset import cifar
from al.model.model_zoo.image_classification import mobilenet
from al.model.cifar import CifarLearner
from al.train.active_train import ActiveTrain
from al.helpers.logger import setup_logger
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


DATASET = 'cifar'

FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(
    __file__, FOLDER_PATH, logging_lvl=10)

logger.info('-------------------------')
logger.info('--LAUNCHING EXPERIMENTS--')
logger.info('-------------------------')

config = load_config(FOLDER_PATH, DATASET)
setupper = set_up_learner(DATASET)

config['active_learning']['output_dir'] = OUTPUT_DIR
config['experiment']['logger_name'] = logger_name
model_name = config['experiment']['model']

score_data = {}

for i in range(config['experiment']['repeats']):
    logger.info('---------------------------')
    logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
    logger.info('---------------------------')
    for strategy in config['experiment']['strategies']:
        dataset, learner = setupper(
            config, OUTPUT_DIR, logger, queries_name=f'queries-{strategy}-{i}-{model_name}.txt')
        logger.info('---------------------------')
        logger.info(f'----STRATEGY : {strategy}----')
        logger.info('---------------------------')
        trainer = ActiveTrain(learner, dataset, strategy,
                              logger_name, strategy_params={'beta': 10})
        scores = trainer.train(
            config['train_parameters'], **config['active_learning'])
        score_data[(strategy, i)] = scores
        logger.info(f'----DONE----\n')
    logger.info('---------------------------')
    logger.info(f'--------DONE--------')
    logger.info('---------------------------\n\n\n')

if config['experiment']['save_results']:
    with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'wb') as f:
        pickle.dump(score_data, f)
