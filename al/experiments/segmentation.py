import numpy as np
from ptsemseg.models import get_model

from ..model.model_zoo import *
from ..model.unet import SemanticLearner
from ..model.configs import cfg
from ..dataset.pascal_voc import PascalVOCSemanticDataset


def set_up_pascalvoc_segmentation(config, output_dir, logger, device=0, queries_name='queries.txt'):
    logger.info('Setting up datasets...')

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    index_test = np.arange(config['dataset']['test_size'])
    logger_name = config['experiment']['logger_name']

    dataset = PascalVOCSemanticDataset(
        index_train, n_init=init_size, output_dir=output_dir, queries_name=queries_name)
    test_dataset = PascalVOCSemanticDataset(
        index_test, n_init=init_size, output_dir=output_dir, train=False, queries_name=queries_name)
    dataset.set_validation_dataset(test_dataset.dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(
        f'Dataset initial test size : {len(test_dataset.init_dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset.dataset)}')

    logger.info('Setting up models...')

    n_classes = dataset.dataset.dataset.n_classes
    model = get_model_config(config, n_classes)
    learner = SemanticLearner(
        model=model, cfg=cfg, logger_name=logger_name, config=config, device=device)
    return dataset, learner


def get_model_config(config, n_classes):
    model = get_model(config['model'], n_classes)
    return model
