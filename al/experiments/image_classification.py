import numpy as np

from al.model.model_zoo import *
from al.model.mnist import MnistLearner
from al.model.cifar import CifarLearner
from al.dataset.mnist import MnistDataset
from al.dataset.cifar import Cifar100Dataset


def set_up_mnist(config, output_dir, logger):
    train_size, val_size, init_size = config['dataset']['train_size'], config['dataset']['val_size'], config['active_learning']['init_size']
    index_validation = np.arange(val_size)
    index_train = np.arange(val_size, train_size+val_size)
    logger.info('Setting up datasets...')

    dataset = MnistDataset(index_train, n_init=init_size, output_dir=output_dir)
    dataset.set_validation_dataset(dataset.get_dataset(index_validation))

    logger.info('Setting up models...')

    model = ConvModel()
    learner = MnistLearner(model)
    return dataset, learner


def set_up_cifar(config, output_dir, logger):
    logger.info('Setting up datasets...')

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    logger_name = config['experiment']['logger_name']

    dataset = Cifar100Dataset(index_train, n_init=init_size, output_dir=output_dir)
    test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.debug(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.debug(f'Dataset used train size : {len(dataset.dataset)}')
    logger.debug(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    model = mobilenet
    learner = CifarLearner(model, cifar100=True, logger_name=logger_name)
    return dataset, learner