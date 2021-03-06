import numpy as np

from al.model.model_zoo import *
from al.model.mnist import MnistLearner
from al.model.cifar import CifarLearner
from al.dataset.mnist import MnistDataset
from al.dataset.cifar import CifarDataset


def set_up_mnist(config, output_dir, logger, device=0, queries_name='queries.txt', index_train=None, index_validation=None):
    train_size, val_size, init_size = config['dataset']['train_size'], config[
        'dataset']['val_size'], config['active_learning']['init_size']
    if index_validation is None:
        index_validation = np.arange(val_size)
    if index_train is None:
        index_train = np.arange(train_size)
    logger_name = config['experiment']['logger_name']
    logger.info('Setting up datasets...')

    dataset = MnistDataset(index_train, n_init=init_size,
                           output_dir=output_dir, queries_name=queries_name)
    test_dataset = MnistDataset(index_validation, n_init=init_size,
                                output_dir=output_dir, queries_name=queries_name, train=False)
    dataset.set_validation_dataset(test_dataset.get_dataset(index_validation))

    logger.info('Setting up models...')
    n_classes = config['experiment'].get('n_class', 10)
    if 'simple_cnn' in config['experiment']['model']:
        model = ConvModel(classes=n_classes)
    elif 'simplenet' in config['experiment']['model']:
        model = simplenet(classes=n_classes)
    learner = MnistLearner(model, logger_name=logger_name, device=device)
    return dataset, learner


def set_up_cifar(config, output_dir, logger, device=0, queries_name='queries.txt'):
    logger.info('Setting up datasets...')

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    logger_name = config['experiment']['logger_name']

    dataset = CifarDataset(
        index_train, n_init=init_size, output_dir=output_dir, queries_name=queries_name, size=config['experiment']['size'])
    test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.debug(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.debug(f'Dataset used train size : {len(dataset.dataset)}')
    logger.debug(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    if 'mobilenet' in config['experiment']['model']:
        model = mobilenet_v2(config)
    elif 'nasnet' in config['experiment']['model']:
        model = nasnet(config)
    learner = CifarLearner(model, cifar100=config['experiment']['size'] == 100,
                           logger_name=logger_name, device=device)
    return dataset, learner
