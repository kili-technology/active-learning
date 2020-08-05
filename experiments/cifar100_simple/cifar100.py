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
REPEATS = 2

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


# TRAIN_SIZE = 30000

# EXPERIMENT_NAME = 'cifar100_simple'
# OUTPUT_DIR = f'experiments/{EXPERIMENT_NAME}/results'
# FIGURE_DIR = f'experiments/{EXPERIMENT_NAME}/figures'
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(FIGURE_DIR, exist_ok=True)

# logger_name = EXPERIMENT_NAME
# logger = setup_logger(logger_name, OUTPUT_DIR)
# logger.setLevel(logging.DEBUG)

# logger.info('Launching simple experiments on CIFAR100')

# experiment_parameters = {
#     'n_repeats': 1,
#     'strategies': ['random_sampling', 'margin_sampling']
# }

# active_parameters = {
#     'assets_per_query': 1000,
#     'n_iter': 15,
#     'init_size': 5000,
#     'compute_score': True,
#     'score_on_train': False,
#     'output_dir': OUTPUT_DIR
# }

# train_parameters = {
#     'batch_size': 32,
#     'iterations': 2000,
#     'learning_rate': 0.003,
#     'shuffle': True
# }

# index_train = np.arange(TRAIN_SIZE)

# def set_up():
#     logger.info('Setting up datasets...')

#     dataset = cifar.Cifar100Dataset(index_train, n_init=active_parameters['init_size'], output_dir=active_parameters['output_dir'])
#     test_dataset = dataset._get_initial_dataset(train=False)
#     dataset.set_validation_dataset(test_dataset)

#     logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
#     logger.info(f'Dataset used train size : {len(dataset.dataset)}')
#     logger.info(f'Dataset test size : {len(test_dataset)}')

#     logger.info('Setting up models...')

#     model = mobilenet
#     learner = CifarLearner(model, cifar100=True, logger_name=logger_name)
#     return dataset, learner


# logger.info('Launching trainings...')

# dataset, learner = set_up()
# strategy='entropy_sampling'
# trainer = ActiveTrain(learner, dataset, strategy, logger_name)
# scores = trainer.train(train_parameters, **active_parameters)

# score_data = {}

# for i in range(experiment_parameters['n_repeats']):
#     logger.info('---------------------------')
#     logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
#     logger.info('---------------------------')
#     for strategy in experiment_parameters['strategies']:
#         dataset, learner = set_up()
#         logger.info('---------------------------')
#         logger.info(f'----STRATEGY : {strategy}----')
#         logger.info('---------------------------')
#         trainer = ActiveTrain(learner, dataset, strategy, logger_name)
#         scores = trainer.train(train_parameters, **active_parameters)
#         score_data[(strategy, i)] = scores
#         logger.info(f'----DONE----\n')
#     logger.info('---------------------------')
#     logger.info(f'--------DONE--------')
#     logger.info('---------------------------\n\n\n')

# with open(f'{OUTPUT_DIR}/scores.pickle', 'wb') as f:
#     pickle.dump(score_data, f)