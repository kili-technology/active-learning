import os
import logging

from ..algorithms import get_strategy
from ..helpers.time import timeit


class ActiveTrain():

    def __init__(self, learner, dataset, method, logger_name=None, strategy_params={}):
        self.learner = learner
        self.dataset = dataset
        self.method = method
        self.strategy = get_strategy(method, logger_name=logger_name, labeled_ds=dataset.get_labeled(), **strategy_params)
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.strategy_params = strategy_params

    @timeit
    def train_iter(self, *args, **kwargs):
        labeled_dataset = self.dataset.get_labeled()
        self.logger.debug(f'Training on {len(labeled_dataset)} samples...')
        self.learner.fit(labeled_dataset, *args, **kwargs)

    @timeit
    def score(self, on_train=False, log_time={}):
        self.logger.debug(f'Scoring...')
        validation_dataset = self.dataset.get_validation_dataset()
        self.logger.debug(f'Computing validation score on {len(validation_dataset)} samples...')
        validation_score = self.learner.score(validation_dataset, batch_size=self.learner.val_batch_size)
        scores = {'val': validation_score}
        if on_train:
            labeled_dataset = self.dataset.get_labeled()
            self.logger.debug(f'Computing training score on {len(labeled_dataset)} samples...')
            train_score = self.learner.score(labeled_dataset, batch_size=self.learner.val_batch_size)
            scores['train'] = train_score
        self.logger.debug(f'Scored.')
        return scores

    @timeit
    def add_to_labeled(self, n, log_time={}):
        self.logger.debug(f'Adding {n} samples to the dataset...')
        unlabeled_dataset = self.dataset.get_unlabeled()
        self.strategy = get_strategy(self.method, logger_name=self.logger_name, labeled_ds=self.dataset.get_labeled(), **self.strategy_params)
        if len(unlabeled_dataset) > 0:
            unlabeled_indices_to_add = self.strategy.return_top_indices(unlabeled_dataset, self.learner, n, log_time=log_time)
            indices_to_add = [self.dataset.unlabeled_to_all[i] for i in unlabeled_indices_to_add]
            self.dataset.add_to_labeled(indices_to_add)
        else:
            indices_to_add = []
        self.logger.debug(f'Added {len(indices_to_add)} samples to the dataset.')
        self.logger.debug(f'Labeled dataset size : {len(self.dataset.get_labeled())}')
        self.logger.debug(f'Unlabeled dataset size : {len(self.dataset.get_unlabeled())}')

    def train(self, train_parameters, n_iter, assets_per_query, compute_score, score_on_train, output_dir, *args, **kwargs):
        list_scores = []
        self.learner.val_batch_size = train_parameters.get('val_batch_size', 64)
        self.logger.info(f'Training for {n_iter} steps, querying {assets_per_query} assets per step.')
        for i in range(n_iter):
            # print(f'Step number #{i+1}')
            log_time = {}
            self.logger.info(f'Step number #{i+1}')
            self.logger.debug(f'Beginning training iteration...')
            train_parameters['log_time'] = log_time
            self.train_iter(**train_parameters)
            self.logger.debug(f'Done')
            if compute_score:
                scores = self.score(on_train=score_on_train, log_time=log_time)
            else:
                scores = {}
            self.logger.debug(f'Beginning active learning query...')
            self.add_to_labeled(assets_per_query, log_time=log_time)
            scores['step'] = i+1
            scores['times'] = log_time
            list_scores.append(scores)
            self.logger.debug('Done.')
            self.logger.info(f'Metrics : {scores}\n')
        return list_scores