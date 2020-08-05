import os
import yaml

from .logger import setup_logger


def set_up_experiment(experiment_name):
    OUTPUT_DIR = f'experiments/{experiment_name}/results'
    FIGURE_DIR = f'experiments/{experiment_name}/figures'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    logger_name = experiment_name
    logger = setup_logger(logger_name, OUTPUT_DIR)
    return OUTPUT_DIR, FIGURE_DIR, logger, logger_name


def load_config(folder, dataset):
    if dataset == 'mnist':
        config_path = os.path.join(folder, 'mnist.yaml')
    elif dataset == 'cifar':
        config_path = os.path.join(folder, 'cifar.yaml')
    elif dataset == 'pascalvoc_detection':
        config_path = os.path.join(folder, 'pascalvoc_detection.yaml')
    elif dataset == 'pascalvoc_segmentation':
        config_path = os.path.join(folder, 'pascalvoc_segmentation.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
