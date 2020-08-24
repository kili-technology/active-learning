import logging

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.optimizers import get_optimizer
from ptsemseg.metrics import runningScore

from .active_model import ActiveLearner
from .model_zoo.ssd import *
from ..helpers.time import timeit
from ..helpers.samplers import IterationBasedBatchSampler


class SemanticLearner(ActiveLearner):

    def __init__(self, model, cfg, logger_name=None, device=0, config=None):
        super().__init__(device=device)
        self.cfg = cfg
        self.model = model
        if self.cuda_available:
            self.model.cuda()

        self.criterion = get_loss_function(config)
        optimizer_cls = get_optimizer(config)
        optimizer_params = {
            k: v for k, v in config["training"]["optimizer"].items() if k != "name"}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)

        self.logger = logging.getLogger(logger_name)
        self.inference_img_size = 16
        self.reducer = nn.AdaptiveAvgPool2d(self.inference_img_size)

    def get_predictions(self, dataset):
        self.model.eval()
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=16, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler)
        predictions = []
        with torch.no_grad():
            for (images, _) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                if self.cuda_available:
                    images = images.cuda()
                masks = self.model(images)
                if self.cuda_available:
                    masks = masks.detach().cpu()
                reduced_masks = self.reducer(masks)
                predictions.append(reduced_masks.data)
        return torch.cat(predictions).numpy()

    def inference(self, dataset):
        self.model.eval()
        predictions = self.get_predictions(dataset)
        probabilities = nn.Softmax2d()(torch.from_numpy(predictions)).numpy()
        return {'predictions': predictions, 'class_probabilities': probabilities}

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
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle), batch_size=batch_size, drop_last=False)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=iterations, start_iter=0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.DATA_LOADER.NUM_WORKERS)
        for step, (images, label_image) in tqdm.tqdm(
                enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
            if self.cuda_available:
                images = images.cuda()
                label_image = label_image.cuda()
            self.model.zero_grad()
            mask_preds = self.model(images)
            self.optimizer.zero_grad()
            loss = self.criterion(input=mask_preds, target=label_image)
            loss.backward()
            self.optimizer.step()

    def score(self, dataset, batch_size, *args, **kwargs):
        self.model.eval()
        running_metrics_val = runningScore(
            dataset.dataset.n_classes)
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler)
        with torch.no_grad():
            for (images, labels_val) in tqdm.tqdm(loader, disable=self.logger.level > 15):
                if self.cuda_available:
                    images = images.cuda()
                    labels_val = labels_val.cuda()
                outputs = self.model(images)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                running_metrics_val.update(gt, pred)
        score, class_iou = running_metrics_val.get_scores()
        return {**score, **class_iou}