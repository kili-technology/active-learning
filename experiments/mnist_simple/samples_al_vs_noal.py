import os
import pickle
import yaml

import tqdm
import numpy as np
import pandas as pd
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import seaborn as sns
from al.helpers.experiment import *
from al.experiments import set_up_learner
from al.helpers.logger import setup_logger


EXPERIMENT_NAME = os.path.dirname(__file__)
model_name = 'simple_cnn'
OUTPUT_DIR = f'{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'{EXPERIMENT_NAME}/figures'
plot_dir = os.path.join(os.path.dirname(__file__), 'figures')

with open(f'{OUTPUT_DIR}/scores-{model_name}.pickle', 'rb') as f:
    scores = pickle.load(f)


df = extract_df(scores)


df_random = extract_strategy(df, 'random_sampling')
df_al = extract_strategy(df, 'margin_sampling')


plot_size_required(df_al, df_random, plot_dir, points=[
                   0.76, 0.8, 0.9, 0.92, 0.95])
