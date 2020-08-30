import logging
import os

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from .active_model import ActiveLearner
from .model_zoo.ssd import BatchCollator, voc_evaluation, coco_evaluation, make_lr_scheduler
from ..helpers.time import timeit
from ..helpers.samplers import IterationBasedBatchSampler


class SSDLearner(ActiveLearner):

    def __init__(self, model, cfg, logger_name=None, device=0, dataset='voc'):
        super().__init__(device=device)
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger(logger_name)
        self.dataset = dataset

    def get_predictions(self, dataset):
        self.model.eval()
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=self.val_batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train=False))
        detections = []
        loader_ids = []
        with torch.no_grad():
            for (images, labels, id_) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                loader_ids += list(id_.numpy())
                if self.cuda_available:
                    images = images.cuda()
                features = self.model.backbone(images)
                cls_logits, bbox_pred = self.model.box_head.predictor(features)
                detection_batch, _ = self.model.box_head._forward_active(
                    cls_logits, bbox_pred)
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

    def send_container_to_cuda(self, targets):
        try:
            if self.cuda_available:
                for key, val in targets._data_dict.items():
                    targets[key] = val.cuda()
            return targets
        except:
            return targets

    def send_container_to_cpu(self, targets):
        if self.cuda_available:
            for key, val in targets._data_dict.items():
                try:
                    targets[key] = val.cpu()
                except:
                    targets[key] = val
        return targets

    @timeit
    def fit(self, dataset, batch_size, learning_rate, momentum, weight_decay, iterations, shuffle=True, *args, **kwargs):
        if self.cuda_available:
            self.model.cuda()
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = make_lr_scheduler(optimizer)
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle), batch_size=batch_size, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=iterations, start_iter=0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train=True))
        for step, (images, targets, _) in tqdm.tqdm(
                enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
            if self.cuda_available:
                images = images.cuda()
                targets = self.send_container_to_cuda(targets)
            self.model.zero_grad()
            loss_dict = self.model(images, targets=targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    def score(self, dataset, batch_size=64, *args, **kwargs):
        self.model.eval()
        results_dict = {}
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train=False))
        with torch.no_grad():
            for (images, targets, image_ids) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                if self.cuda_available:
                    images = images.cuda()
                outputs = self.model(images)
                results_dict.update(
                    {img_id: self.send_container_to_cpu(
                        result) for img_id, result in zip(image_ids, outputs)}
                )
        if self.dataset == 'voc':
            return voc_evaluation(dataset=dataset, predictions=results_dict, output_dir=None, iteration=None)
        elif self.dataset == 'coco':
            return coco_evaluation(dataset=dataset, predictions=results_dict, output_dir=os.path.expanduser('~/data/coco'), iteration=None)
