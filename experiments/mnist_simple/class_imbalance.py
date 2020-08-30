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

strategies = ['random_sampling', 'margin_sampling']
repeats = 1
score_data = {}
config['active_learning']['assets_per_query'] = 20
config['active_learning']['n_iter'] = 5
config['active_learning']['init_size'] = 100

config['train_parameters']['batch_size'] = 16
config['train_parameters']['iterations'] = 100

config['experiment']['n_classes'] = 2

raw_dataset, _ = setupper(config, OUTPUT_DIR, logger,
                          index_train=np.arange(60000))
full_train_dataset = raw_dataset.dataset

first_class = 1
second_class = 2
first_classes = []
second_classes = []
p = 0.1

for i in range(len(full_train_dataset)):
    if full_train_dataset[i][1].numpy() == first_class:
        first_classes.append(i)
    elif full_train_dataset[i][1].numpy() == second_class and np.random.rand() < p:
        second_classes.append(i)

train_indices = np.array(first_classes + second_classes)
np.random.permutation(train_indices)

for i in range(repeats):
    logger.info('---------------------------')
    logger.info(f'--------ROUND OF TRAININGS NUMBER #{i+1}--------')
    logger.info('---------------------------')
    for strategy in strategies:
        dataset, learner = setupper(
            config, OUTPUT_DIR, logger, index_train=train_indices)
        logger.info('---------------------------')
        logger.info(f'----STRATEGY : {strategy}----')
        logger.info('---------------------------')
        trainer = ActiveTrain(learner, dataset, strategy, logger_name)
        scores = trainer.train(
            config['train_parameters'], **config['active_learning'])
        score_data[(strategy, i)] = scores
        logger.info(f'----DONE----\n')
    logger.info('---------------------------')
    logger.info(f'--------DONE--------')
    logger.info('---------------------------\n\n\n')


# data = []
# for (strategy, experiment_number), scores_experiment in score_data.items():
#     for step_result in scores_experiment:
#         val_step_result = step_result['val']
#         step = step_result['step']
#         data.append(
#             {'strategy': strategy,
#              'experiment': experiment_number,
#              'step': step,
#              **val_step_result})
# df = pd.DataFrame(data)

# plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

# plt.figure(num=0, figsize=(12, 5))
# sns.lineplot(x='step', y='accuracy', hue='strategy', data=df)
# plt.ylabel('Accuracy')
# plt.show()
# plt.savefig(os.path.join(plot_dir, 'accuracy_imbalance.png'))
