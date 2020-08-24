import os
import logging
import pickle

import numpy as np

import al
from al.dataset import pascal_voc
from al.dataset import active_dataset
from al.model.model_zoo.unet import UNet
from al.model.model_zoo.image_classification import mobilenet
from al.model.ssd import SSDLearner
from al.model.configs import cfg
from al.train.active_train import ActiveTrain
from al.helpers.experiment import set_up_experiment, load_config
from al.experiments import set_up_learner


DATASET = 'pascalvoc_segmentation'
FOLDER_PATH = os.path.dirname(__file__)

OUTPUT_DIR, FIGURE_DIR, logger, logger_name = set_up_experiment(__file__, FOLDER_PATH, logging_lvl=10)


logger.info('-------------------------')
logger.info('--LAUNCHING EXPERIMENTS--')
logger.info('-------------------------')

config = load_config(FOLDER_PATH, DATASET)
setupper = set_up_learner(DATASET)

config['active_learning']['output_dir'] = OUTPUT_DIR
config['experiment']['logger_name'] = logger_name
logger.debug('Getting dataset and learner')
dataset, learner = setupper(config, OUTPUT_DIR, logger, device=1)
logger.debug('Getting trainer')
trainer = ActiveTrain(learner, dataset, config['experiment']['strategies'][0], logger_name)
logger.debug('Training...')

scores = trainer.train(config['train_parameters'], **config['active_learning'])
logger.debug('Done training...')

# index_train = np.arange(TRAIN_SIZE)

# config_file = 'al/model/configs/unet.yaml'

# def get_model_config(config_file):
#     cfg.merge_from_file(config_file)
#     if 'unet' in config_file:
#         model = UNet(cfg)
#     cfg.freeze()
#     return model, cfg


# def set_up():
#     logger.info('Setting up datasets...')
#     model, cfg = get_model_config(config_file)

#     dataset = pascal_voc.PascalVOCSemanticDataset(index_train, n_init=active_parameters['init_size'], cfg=cfg)
#     # test_dataset = active_dataset.MaskDataset(dataset._get_initial_dataset(train=False), np.arange(40))
#     test_dataset = dataset._get_initial_dataset(train=False)
#     dataset.set_validation_dataset(test_dataset)

#     logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
#     logger.info(f'Dataset used train size : {len(dataset.dataset)}')
#     logger.info(f'Dataset test size : {len(test_dataset)}')

#     logger.info('Setting up models...')

#     img, label = dataset.dataset[0]

#     print(img.shape, label.shape)
#     print(np.unique(label))

    
#     learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name)
#     return dataset, learner

# logger.info('Launching trainings...')

# dataset, learner = set_up()

# # strategy='al_for_deep_object_detection'
# # strategy='random_sampling'
# # trainer = ActiveTrain(learner, dataset, strategy, logger_name)
# # scores = trainer.train(train_parameters, **active_parameters)
