import logging

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from .active_model import ActiveLearner
from ..helpers.time import timeit
from ..helpers.samplers import IterationBasedBatchSampler
from ..model.utils import enable_dropout, disable_dropout


class MnistLearner(ActiveLearner):

    def __init__(self, model, logger_name=None, device=0):
        super().__init__(device=device)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(logger_name)

    def get_predictions(self, dataset, bayesian=False):
        self.model.eval()
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=256, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler)
        preds = []
        list_features = []
        if bayesian:
            enable_dropout(self.model)
        with torch.no_grad():
            for step, (data, _) in tqdm.tqdm(
                    enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
                if data.dim() != 4:
                    data = data[:, None, :, :]
                if self.cuda_available:
                    data = data.cuda()
                prediction, features = self.model(data, features=True)
                if self.cuda_available:
                    prediction = prediction.detach().cpu()
                    features = features.detach().cpu()
                preds.append(prediction.data)
                list_features.append(features.data)
        return (torch.cat(preds).numpy(), torch.cat(list_features).numpy())

    def inference(self, dataset, bayesian=False):
        predictions, features = self.get_predictions(
            dataset, bayesian=bayesian)
        probabilities = np.exp(predictions) / \
            np.exp(predictions).sum(axis=1)[:, None]
        return {'class_probabilities': probabilities, 'predictions': predictions, 'features': features}

    @staticmethod
    def get_base_sampler(size, shuffle):
        if shuffle:
            order = np.random.permutation(np.arange(size))
            return SequentialSampler(order)
        else:
            return SequentialSampler(range(size))

    @timeit
    def fit(self, dataset, batch_size, learning_rate, iterations, shuffle=True, *args, **kwargs):
        self.model.train()
        disable_dropout(self.model)
        if self.cuda_available:
            self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle), batch_size=batch_size, drop_last=False)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=iterations, start_iter=0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler)
        for step, (data, targets) in tqdm.tqdm(
                enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
            self.model.zero_grad()
            if data.dim() != 4:
                data = data[:, None, :, :]
            if self.cuda_available:
                data = data.cuda()
                targets = targets.cuda()
            prediction = self.model(data)
            loss = self.criterion(prediction, targets)
            loss.backward()
            optimizer.step()

    def score(self, dataset, batch_size=256, *args, **kwargs):
        self.model.eval()
        disable_dropout(self.model)
        total_accuracy = 0.0
        total_loss = 0.0
        batch_sampler = BatchSampler(
            sampler=self.get_base_sampler(len(dataset), shuffle=False), batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler)
        with torch.no_grad():
            for step, (data, targets) in tqdm.tqdm(
                    enumerate(loader), disable=self.logger.level > 15, total=len(loader)):
                if data.dim() != 4:
                    data = data[:, None, :, :]
                if self.cuda_available:
                    data = data.cuda()
                    targets = targets.cuda()
                prediction = self.model(data)
                total_loss += self.criterion(prediction,
                                             targets).item() * data.size(0)
                _, number_predicted = torch.max(prediction.data, 1)
                total_accuracy += (number_predicted == targets).sum()
        accuracy = float(
            (total_accuracy / len(dataset)).detach().cpu().numpy())
        mean_loss = (total_loss / len(dataset))
        return {'accuracy': accuracy, 'loss': mean_loss}
