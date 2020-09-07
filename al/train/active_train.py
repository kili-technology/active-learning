"""
Base class for training an active learner (see models/) and an active dataset (see dataset/)
"""

import os
import logging

from ..algorithms import get_strategy
from ..helpers.time import timeit


class ActiveTrain():

    def __init__(self, learner, dataset, method, logger_name=None, strategy_params={}):
        """
        Initializes a learner, a dataset, and a method (string)
        """
        self.learner = learner
        self.dataset = dataset
        self.method = method
        self.strategy = get_strategy(
            method, logger_name=logger_name, labeled_ds=dataset.get_labeled(), **strategy_params)
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.strategy_params = strategy_params

    @timeit
    def train_iter(self, *args, **kwargs):
        """
        Fit the learner on the labeled dataset

        Parameters
        ----------
        - *args : given to the fit method of the learner
        - **kwargs : given to the fit method of the learner
        """
        labeled_dataset = self.dataset.get_labeled()
        size_labeled = len(labeled_dataset)
        self.logger.debug(f'Training on {size_labeled} samples...')
        metrics_train = self.learner.fit(labeled_dataset, *args, **kwargs)
        if metrics_train is None:
            metrics_train = {}
        return size_labeled, metrics_train

    @timeit
    def score(self, on_train=False, log_time={}):
        """
        Score the learner on the validation dataset of the dataset object.

        Parameters
        ----------
        - on_train : bool. If True, the learner will also be evaluated on the training dataset.
        """
        self.logger.debug(f'Scoring...')
        validation_dataset = self.dataset.get_validation_dataset()
        self.logger.debug(
            f'Computing validation score on {len(validation_dataset)} samples...')
        validation_score = self.learner.score(
            validation_dataset, batch_size=self.learner.val_batch_size)
        scores = {'val': validation_score}
        if on_train:
            labeled_dataset = self.dataset.get_labeled()
            self.logger.debug(
                f'Computing training score on {len(labeled_dataset)} samples...')
            train_score = self.learner.score(
                labeled_dataset, batch_size=self.learner.val_batch_size)
            scores['train'] = train_score
        self.logger.debug(f'Scored.')
        return scores

    @timeit
    def add_to_labeled(self, n, log_time={}):
        """
        Active learning query. Selects which unlabeled samples to add to the training data.

        Parameters
        ----------
        - n : int. Number of samples to select and add to the training dataset.
        """
        self.logger.debug(f'Adding {n} samples to the dataset...')
        unlabeled_dataset = self.dataset.get_unlabeled()
        self.strategy = get_strategy(self.method, logger_name=self.logger_name,
                                     labeled_ds=self.dataset.get_labeled(), **self.strategy_params)
        if len(unlabeled_dataset) > 0:
            unlabeled_indices_to_add = self.strategy.return_top_indices(
                unlabeled_dataset, self.learner, n, log_time=log_time)
            indices_to_add = [self.dataset.unlabeled_to_all[i]
                              for i in unlabeled_indices_to_add]
            self.dataset.add_to_labeled(indices_to_add)
        else:
            indices_to_add = []
        self.logger.debug(
            f'Added {len(indices_to_add)} samples to the dataset.')
        self.logger.debug(
            f'Labeled dataset size : {len(self.dataset.get_labeled())}')
        self.logger.debug(
            f'Unlabeled dataset size : {len(self.dataset.get_unlabeled())}')

    @timeit
    def active_iter(self, train_parameters, n_iter, assets_per_query, compute_score, score_on_train, output_dir, step, log_time={}):
        """
        Single active learning training iteration : fitting a model, computing score, and running the AL algorithm
        on the unlabeled dataset to add samples to the labeled dataset.

        Parameters
        ----------
        - train_parameters : dict. Parameters given to the fit method of the learner
        - n_iter : int. Number of active learning training iterations
        - assets_per_query : int. Number of samples the active learning algorithm queries in this iteration
        - compute_score : bool. If True, computes score on validation dataset
        - score_on_train : bool. If True, computes score on training dataset
        - output_dir : str. Path to directory where possible outputs are written
        - step : current iteration of the training.

        Returns
        -------
        - scores : dict. Metrics and informations on the iteration.
        """
        self.logger.info(f'Step number #{step}')
        self.logger.debug(f'Beginning training iteration...')
        train_parameters['log_time'] = log_time
        size_labeled, metrics_train = self.train_iter(**train_parameters)
        self.logger.debug(f'Done')
        if compute_score:
            scores = self.score(on_train=score_on_train, log_time=log_time)
        else:
            scores = {}
        self.logger.debug(f'Beginning active learning query...')
        self.add_to_labeled(assets_per_query, log_time=log_time)
        scores['step'] = step
        scores['size_labeled'] = size_labeled
        scores['train_metrics'] = metrics_train
        self.logger.debug('Done.')
        return scores

    def train(self, train_parameters, n_iter, assets_per_query=100, compute_score=True, score_on_train=False, output_dir='', *args, **kwargs):
        """
        Whole active learning training

        Parameters
        ----------
        - train_parameters : dict. Parameters given to the fit method of the learner
        - n_iter : int. Number of active learning training iterations
        - assets_per_query : int. Number of samples the active learning algorithm queries in this iteration
        - compute_score : bool. If True, computes score on validation dataset
        - score_on_train : bool. If True, computes score on training dataset
        - output_dir : str. Path to directory where possible outputs are written
        """
        list_scores = []
        self.learner.val_batch_size = train_parameters.get(
            'val_batch_size', 256)
        self.logger.info(
            f'Training for {n_iter} steps, querying {assets_per_query} assets per step.')
        for i in range(n_iter):
            log_time = {}
            scores = self.active_iter(
                train_parameters, n_iter, assets_per_query, compute_score, score_on_train, output_dir, i+1, log_time=log_time)
            scores['times'] = log_time
            self.logger.info(f'Metrics : {scores}\n')
            list_scores.append(scores)
        return list_scores
