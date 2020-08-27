import os
import logging
import pickle

import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


config = load_config(FOLDER_PATH, DATASET)
setupper = set_up_learner(DATASET)
config['active_learning']['output_dir'] = OUTPUT_DIR
config['experiment']['logger_name'] = logger_name
model_name = config['experiment']['model']
dataset, learner = setupper(config, OUTPUT_DIR, logger)

queried = os.path.join(os.path.dirname(__file__), 'results',
                       'queries-margin_sampling-0-simplenet.txt')
df = pd.read_csv(queried, header=0, skiprows=1)
# print(df)
query_step = 0
plot_size = 32
indices = df.loc[query_step].values

train_dataset = dataset.dataset
tensor = torch.stack([
    train_dataset[i][0].unsqueeze(0) for i in indices
])[:plot_size]

print(tensor.shape)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


plot_dir = os.path.join(os.path.dirname(__file__), 'figures')
plt.figure(figsize=(20, 10))
show(torchvision.utils.make_grid(tensor, nrow=8))
# plt.title("", fontsize=14)
plt.tight_layout()
plt.axis('off')
plt.show()
plt.savefig(os.path.join(plot_dir, 'samples.png'), dpi=200)
