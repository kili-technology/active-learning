import logging

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from .active_model import ActiveLearner
from .model_zoo.ssd import *
from ..helpers.time import timeit
from ..helpers.samplers import IterationBasedBatchSampler


class UNetLearner(ActiveLearner):

    def __init__(self, model, cfg, logger_name=None):
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(logger_name)

    def get_predictions(self, dataset):
        self.model.eval()
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=16, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY)
        detections = []
        loader_ids = []
        with torch.no_grad():
            for (images, labels, id_) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                loader_ids += list(id_.numpy())
                features = self.model.backbone(images)
                cls_logits, bbox_pred = self.model.box_head.predictor(features)
                detection_batch, _ = self.model.box_head._forward_active(cls_logits, bbox_pred)
                detections += detection_batch
        return detections, loader_ids

    def inference(self, dataset):
        self.model.eval()
        detections, loader_ids = self.get_predictions(dataset)
        return {'detections': detections, 'image_ids': loader_ids}

    @staticmethod
    def get_base_sampler(size, shuffle):
        if shuffle:
            order = np.random.permutation(np.arange(size))
            return SequentialSampler(order)
        else:
            return SequentialSampler(range(size))

    @timeit
    def fit(self, dataset, batch_size, learning_rate, momentum, weight_decay, iterations, shuffle=True, *args, **kwargs):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle), batch_size=batch_size, drop_last=False)
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=iterations, start_iter=0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollatorSemantic(is_train=True))
        for step, (images, label_image) in tqdm.tqdm(
                enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
            self.model.zero_grad()
            print(images.shape, label_image.shape)
            mask_preds = self.model(images)
            print(type(mask_preds), type(label_image))
            print(mask_preds.shape)
            loss = self.criterion(mask_preds, label_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def score(self, dataset, batch_size=16, *args, **kwargs):
        self.model.eval()
        results_dict = {}
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY)
        with torch.no_grad():
            for (images, targets, image_ids) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                outputs = self.model(images)
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, outputs)}
                )
        return voc_evaluation(dataset=dataset, predictions=results_dict, output_dir=None, iteration=None)