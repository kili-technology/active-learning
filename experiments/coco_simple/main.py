import os
import logging
import pickle

import numpy as np

import al
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


DATASET = 'coco_object_detection'

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
model_name = config['model']['backbone']


logger.debug('Getting dataset and learner')
strategy = config['experiment']['strategy']
dataset, learner = setupper(config, OUTPUT_DIR, logger, device=0,
                            queries_name=f'queries-{strategy}-{model_name}.txt')
logger.debug('Getting trainer')
trainer = ActiveTrain(
    learner, dataset, strategy, logger_name)
logger.debug('Training...')

scores = trainer.train(config['train_parameters'], **config['active_learning'])


if config['experiment']['save_results']:
    with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'wb') as f:
        pickle.dump(scores, f)
