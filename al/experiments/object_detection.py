import numpy as np

from ..model.model_zoo import *
from ..model.ssd import SSDLearner
from ..model.unet import UNetLearner
from ..dataset.pascal_voc import PascalVOCObjectDataset, PascalVOCSemanticDataset
from ..model.configs import cfg



def set_up_pascalvoc_detection(config, output_dir, logger, device=0, queries_name='queries.txt'):
    logger.info('Setting up datasets...')
    config_file = config['experiment']['config_file']
    model, cfg = get_model_config(config_file, config['model_type'])

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    index_test = np.arange(config['dataset']['test_size'])
    logger_name = config['experiment']['logger_name']

    dataset = PascalVOCObjectDataset(
        index_train, n_init=init_size, output_dir=output_dir, cfg=cfg)
    # test_dataset = dataset._get_initial_dataset(train=False)
    test_dataset = PascalVOCObjectDataset(
        index_test, n_init=init_size, output_dir=output_dir, cfg=cfg).get_dataset(index_test)
    # test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name, device=device)
    return dataset, learner


def set_up_pascalvoc_segmentation(config, output_dir, logger):
    logger.info('Setting up datasets...')
    config_file = config['experiment']['config_file']
    model, cfg = get_model_config(config_file, config['model_type'])

    init_size = config['active_learning']['init_size']
    index_train = np.arange(config['dataset']['train_size'])
    logger_name = config['experiment']['logger_name']

    dataset = PascalVOCSemanticDataset(
        index_train, n_init=init_size, output_dir=output_dir,  cfg=cfg)
    test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    learner = UNetLearner(model=model, cfg=cfg, logger_name=logger_name)
    return dataset, learner

def get_model_config(config_file, model_type):
    if model_type == 'detection':
        cfg.merge_from_file(config_file)
        if 'mobilenet' in config_file:
            model = SSDDetector(cfg)
        cfg.freeze()
    elif model_type == 'segmentation':
        cfg.merge_from_file(config_file)
        if 'unet' in config_file:
            model = UNet(cfg)
        cfg.freeze()
    return model, cfg