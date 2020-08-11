import os
import logging
import pickle

import numpy as np

import al
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


EXPERIMENT_NAME = 'coco_simple'
DATASET = 'coco_object_detection'

FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(EXPERIMENT_NAME, logging_lvl=10)

logger.info('-------------------------')
logger.info('--LAUNCHING EXPERIMENTS--')
logger.info('-------------------------')

config = load_config(FOLDER_PATH, DATASET)
setupper = set_up_learner(DATASET)

config['active_learning']['output_dir'] = OUTPUT_DIR
config['experiment']['logger_name'] = logger_name
logger.debug('Getting dataset and learner')
dataset, learner = setupper(config, OUTPUT_DIR, logger)
logger.debug('Getting trainer')
trainer = ActiveTrain(learner, dataset, config['experiment']['strategy'], logger_name)
logger.debug('Training...')

scores = trainer.train(config['train_parameters'], **config['active_learning'])


if config['experiment']['save_results']:
    with open(f'{OUTPUT_DIR}/scores.pickle', 'wb') as f:
        pickle.dump(scores, f)