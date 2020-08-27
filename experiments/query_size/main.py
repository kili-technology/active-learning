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
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(
    __file__, FOLDER_PATH, logging_lvl=20)


def experiment_with(dataset_name):
    config = load_config(FOLDER_PATH, dataset_name)
    setupper = set_up_learner(dataset_name)

    config['active_learning']['output_dir'] = OUTPUT_DIR
    config['experiment']['logger_name'] = logger_name
    model_name = config['experiment']['model']
    iterations_per_labeled_sample = config['experiment']['iterations_per_labeled_sample']
    size_to_label = config['experiment']['size_to_label']

    score_data = {}
    logger.info('---------------------------------------')
    logger.info(f'--LAUNCHING EXPERIMENTS ON {dataset_name}--')
    logger.info('---------------------------------------')
    for i in range(config['experiment']['repeats']):
        logger.info('---------------------------')
        logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
        logger.info('---------------------------')
        for query_size in config['experiment']['query_sizes']:
            config['active_learning']['assets_per_query'] = query_size
            config['active_learning']['n_iter'] = np.ceil(
                size_to_label / query_size).astype(int)
            dataset, learner = setupper(
                config, OUTPUT_DIR, logger, queries_name=f'queries-{query_size}-{i}-{model_name}.txt')
            logger.info('---------------------------')
            logger.info(f'----QUERY SIZE : {query_size}----')
            logger.info('---------------------------')
            trainer = ActiveTrain(
                learner, dataset, config['experiment']['strategy'], logger_name)
            scores = trainer.train(
                config['train_parameters'], **config['active_learning'])
            score_data[(query_size, i)] = scores
            logger.info(f'----DONE----\n')
        logger.info('---------------------------')
        logger.info(f'--------DONE--------')
        logger.info('---------------------------\n\n\n')
    if config['experiment']['save_results']:
        with open(f'{OUTPUT_DIR}/scores-{dataset_name}-{model_name}.pickle', 'wb') as f:
            pickle.dump(score_data, f)


if __name__ == '__main__':
    dataset = 'mnist'
    experiment_with(dataset)
