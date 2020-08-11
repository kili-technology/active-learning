import numpy as np

from ..model.model_zoo import *
from ..model.unet import UNetLearner
from ..model.configs import cfg
from ..dataset.pascal_voc import PascalVOCSemanticDataset


def set_up_pascalvoc_segmentation(config, output_dir, logger):
    logger.info('Setting up datasets...')
    model, cfg = get_model_config(None, 'voc')

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



def get_model_config(backbone, dataset):
    if dataset == 'voc':
        config_file = ''
        cfg.merge_from_file(config_file)
        if 'unet' in config_file:
            model = UNet(cfg)