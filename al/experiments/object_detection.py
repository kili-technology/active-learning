import os

import numpy as np

from ..model.model_zoo import *
from ..model.ssd import SSDLearner
from ..dataset.pascal_voc import PascalVOCObjectDataset
from ..dataset.coco import COCOObjectDataset
from ..model.configs import cfg


def set_up_pascalvoc_detection(config, output_dir, logger, device=0, queries_name='queries.txt'):
    logger.info('Setting up datasets...')
    backbone = config['model']['backbone']
    model, cfg = get_model_config(backbone, 'voc')

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    index_test = np.arange(config['dataset']['test_size'])
    logger_name = config['experiment']['logger_name']

    dataset = PascalVOCObjectDataset(
        index_train, n_init=init_size, output_dir=output_dir, cfg=cfg, queries_name=queries_name)
    test_dataset = PascalVOCObjectDataset(
        index_test, n_init=init_size, output_dir=output_dir, cfg=cfg, train=False, queries_name=queries_name)
    dataset.set_validation_dataset(test_dataset.dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset initial test size : {len(test_dataset.init_dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset.dataset)}')

    logger.info('Setting up models...')

    learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name, device=device, dataset='voc')
    return dataset, learner


def set_up_coco_object_detection(config, output_dir, logger, device=0, queries_name='queries.txt'):
    logger.info('Setting up datasets...')
    backbone = config['model']['backbone']
    model, cfg = get_model_config(backbone, 'coco')

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    index_test = np.arange(config['dataset']['test_size'])
    logger_name = config['experiment']['logger_name']

    dataset = COCOObjectDataset(
        index_train, n_init=init_size, output_dir=output_dir, cfg=cfg, queries_name=queries_name)
    test_dataset = COCOObjectDataset(
        index_test, n_init=init_size, output_dir=output_dir, cfg=cfg, train=False, queries_name=queries_name)
    

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset initial test size : {len(test_dataset.init_dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset.dataset)}')
    dataset.set_validation_dataset(test_dataset.dataset)

    logger.info('Setting up models...')

    learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name, device=device, dataset='coco')
    return dataset, learner


def get_model_config(backbone, dataset):
    config_path = os.getenv('MODULE_PATH')
    if dataset == 'voc':
        if backbone == 'mobilenet_v2':
            config_file = 'mobilenet_v2_ssd320_voc0712.yaml'
        elif backbone == 'vgg':
            config_file = 'vgg_ssd300_voc0712.yaml'
    elif dataset == 'coco':
        if backbone == 'vgg':
            config_file = 'vgg_ssd300_coco_trainval35k.yaml'
        elif backbone == 'mobilenet_v2':
            config_file = 'mobilenet_v2_ssd320_coco.yaml'
    path = os.path.expanduser(os.path.join(config_path, config_file))
    cfg.merge_from_file(path)
    model = SSDDetector(cfg, backbone)
    cfg.freeze()
    return model, cfg