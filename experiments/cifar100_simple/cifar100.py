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


EXPERIMENT_NAME = 'cifar100_simple'
DATASET = 'cifar'

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
trainer = ActiveTrain(learner, dataset, config['experiment']['strategies'][0], logger_name)
logger.debug('Training...')

scores = trainer.train(config['train_parameters'], **config['active_learning'])
