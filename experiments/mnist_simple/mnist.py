import os
import logging
import pickle

import numpy as np

import al
from al.dataset import mnist
from al.model.model_zoo.simple_cnn import ConvModel
from al.model.mnist import MnistLearner
from al.dataset.mnist import MnistDataset
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment



VAL_SIZE = 5000
TRAIN_SIZE = 10000
EXPERIMENT_NAME = 'mnist_simple'
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(EXPERIMENT_NAME)


logger.setLevel(logging.DEBUG)
logger.info('Launching simple experiments on Mnist')

experiment_parameters = {
    'save_results': False,
    # 'n_repeats': 5,
    'n_repeats': 1,
    'strategies': ['coreset']
    # 'strategies': ['random_sampling']
    # 'strategies': ['coreset', 'random_sampling', 'uncertainty_sampling', 'margin_sampling', 'entropy_sampling']
}

active_parameters = {
    'assets_per_query': 50,
    'n_iter': 30,
    'init_size': 100,
    'compute_score': True,
    'score_on_train': True,
    'output_dir': OUTPUT_DIR
}

train_parameters = {
    'batch_size': 32,
    'iterations': 100,
    'learning_rate': 0.001,
    'shuffle': True
}

index_validation = np.arange(VAL_SIZE)
index_train = np.arange(VAL_SIZE, TRAIN_SIZE+VAL_SIZE)


def set_up():
    logger.info('Setting up datasets...')

    dataset = mnist.MnistDataset(index_train, n_init=active_parameters['init_size'], output_dir=active_parameters['output_dir'])
    dataset.set_validation_dataset(dataset.get_dataset(index_validation))

    logger.info('Setting up models...')

    model = ConvModel()
    learner = MnistLearner(model, logger_name=logger_name)
    return dataset, learner



# method = 'uncertainty_sampling'
# trainer = ActiveTrain(learner, dataset, method)
# trainer.train(train_parameters, **active_parameters)

logger.info('Launching trainings...')


score_data = {}

for i in range(experiment_parameters['n_repeats']):
    logger.info('---------------------------')
    logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
    logger.info('---------------------------')
    for strategy in experiment_parameters['strategies']:
        dataset, learner = set_up()
        logger.info('---------------------------')
        logger.info(f'----STRATEGY : {strategy}----')
        logger.info('---------------------------')
        trainer = ActiveTrain(learner, dataset, strategy, logger_name)
        scores = trainer.train(train_parameters, **active_parameters)
        score_data[(strategy, i)] = scores
        logger.info(f'----DONE----\n')
    logger.info('---------------------------')
    logger.info(f'--------DONE--------')
    logger.info('---------------------------\n\n\n')

if experiment_parameters['save_results']:
    with open(f'{OUTPUT_DIR}/scores.pickle', 'wb') as f:
        pickle.dump(score_data, f)