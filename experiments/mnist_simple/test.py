import os
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import al
from al.dataset import mnist
from al.model.model_zoo.simple_cnn import ConvModel
from al.model.mnist import MnistLearner
from al.dataset.mnist import MnistDataset
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


DATASET = 'mnist'

FOLDER_PATH = os.path.dirname(__file__)
OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(
    __file__, FOLDER_PATH, logging_lvl=20)

logger.info('-------------------------')
logger.info('--LAUNCHING EXPERIMENTS--')
logger.info('-------------------------')

config = load_config(FOLDER_PATH, DATASET)
setupper = set_up_learner(DATASET)

config['active_learning']['output_dir'] = OUTPUT_DIR
config['experiment']['logger_name'] = logger_name
model_name = 'simple_cnn'
config['experiment']['model'] = model_name

strategies = ['diverse_mini_batch_sampling',
              'random_sampling', 'margin_sampling']
repeats = 1

score_data = {}
config['active_learning']['assets_per_query'] = 100
config['active_learning']['n_iter'] = 9
config['active_learning']['init_size'] = 100

config['train_parameters']['batch_size'] = 16
config['train_parameters']['iterations'] = 200


for i in range(repeats):
    logger.info('---------------------------')
    logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
    logger.info('---------------------------')
    for strategy in strategies:
        if strategy == 'diverse_mini_batch_sampling':
            strategy_params = {'beta': 50}
        else:
            strategy_params = {}
        dataset, learner = setupper(
            config, OUTPUT_DIR, logger)
        logger.info('---------------------------')
        logger.info(f'----STRATEGY : {strategy}----')
        logger.info('---------------------------')
        trainer = ActiveTrain(learner, dataset, strategy,
                              logger_name, strategy_params=strategy_params)
        scores = trainer.train(
            config['train_parameters'], **config['active_learning'])
        score_data[(strategy, i)] = scores
        logger.info(f'----DONE----\n')
    logger.info('---------------------------')
    logger.info(f'--------DONE--------')
    logger.info('---------------------------\n\n\n')


data = []
for (strategy, experiment_number), scores_experiment in score_data.items():
    for step_result in scores_experiment:
        val_step_result = step_result['val']
        step = step_result['step']
        data.append(
            {'strategy': strategy,
             'experiment': experiment_number,
             'step': step,
             **val_step_result})
df = pd.DataFrame(data)

plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

plt.figure(num=0, figsize=(12, 5))
sns.lineplot(x='step', y='accuracy', hue='strategy', data=df)
plt.ylabel('Accuracy')
plt.show()
plt.savefig(os.path.join(plot_dir, 'accuracy_test.png'))
