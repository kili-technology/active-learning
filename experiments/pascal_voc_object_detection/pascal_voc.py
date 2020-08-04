import os
import logging
import pickle

import numpy as np

import al
from al.dataset import pascal_voc
from al.dataset import active_dataset
from al.model.model_zoo.ssd import SSDDetector
from al.model.model_zoo.image_classification import mobilenet
from al.model.ssd import SSDLearner
from al.model.configs import cfg
from al.train.active_train import ActiveTrain
from al.helpers.logger import setup_logger



TRAIN_SIZE = 4000

EXPERIMENT_NAME = 'pascal_voc_object_detection'
OUTPUT_DIR = f'experiments/{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'experiments/{EXPERIMENT_NAME}/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

logger_name = EXPERIMENT_NAME
logger = setup_logger(logger_name, OUTPUT_DIR)
logger.setLevel(logging.DEBUG)

logger.info('Launching simple experiments on Pascal VOC')

experiment_parameters = {
    'n_repeats': 2,
    'strategies': ['random_sampling', 'al_for_deep_object_detection']
}

active_parameters = {
    'assets_per_query': 30,
    'n_iter': 10,
    'init_size': 1000,
    'compute_score': True,
    'score_on_train': False,
    'output_dir': OUTPUT_DIR
}

train_parameters = {
    'batch_size': 16,
    'iterations': 50,
    'learning_rate': 0.001,
    'shuffle': True,
    'momentum': 0.9,
    'weight_decay': 5e-4
}

index_train = np.arange(TRAIN_SIZE)

config_file = 'al/model/configs/mobilenet_v2_ssd320_voc0712.yaml'

def get_model_config(config_file):
    cfg.merge_from_file(config_file)
    if 'mobilenet' in config_file:
        model = SSDDetector(cfg)
    cfg.freeze()
    return model, cfg


def set_up():
    logger.info('Setting up datasets...')
    model, cfg = get_model_config(config_file)

    dataset = pascal_voc.PascalVOCObjectDataset(
        index_train, n_init=active_parameters['init_size'], output_dir=active_parameters['output_dir'], cfg=cfg)
    test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    
    learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name)
    return dataset, learner

logger.info('Launching trainings...')

dataset, learner = set_up()


strategy = 'al_for_deep_object_detection'
# strategy='random_sampling'
trainer = ActiveTrain(learner, dataset, strategy, logger_name,
    strategy_params={'agregation': 'sum', 'weighted': True})
scores = trainer.train(train_parameters, **active_parameters)

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