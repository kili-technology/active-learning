# https://arxiv.org/pdf/1703.02910.pdf, Deep Bayesian Active Learning with Image Data

import numpy as np

from .baseline import Strategy
from ..helpers.time import timeit

class BayesianActiveLearning(Strategy):

    def __init__(self, nb_forward=10, **kwargs):
        super(BayesianActiveLearning, self).__init__()
        self.nb_forward = nb_forward

    @timeit
    def evaluate_dataset(self, dataset, learner, log_time={}):
        return np.stack([learner.inference(dataset, bayesian=True)['class_probabilities'] for _ in range(self.nb_forward)])

    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        raise NotImplementedError

    def return_top_indices(self, dataset, learner, top, log_time={}):
        scores = self.score_dataset(dataset, learner, log_time=log_time)
        sorted_idx = np.argsort(scores)
        return sorted_idx[-top:]


class BayesianEntropyStrategy(BayesianActiveLearning):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        stacked_probabilities = self.evaluate_dataset(dataset, learner, log_time=log_time)
        probabilities = np.mean(stacked_probabilities, axis=0)
        assert len(probabilities) == len(dataset)
        entropies = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropies


class BayesianBALDStrategy(BayesianActiveLearning):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        model_probabilities = inference_result['class_probabilities']
        model_entropies = -np.sum(model_probabilities * np.log(model_probabilities), axis=1)
        stacked_probabilities = self.evaluate_dataset(dataset, learner, log_time=log_time)
        average_probabilities = np.mean(stacked_probabilities, axis=0)
        average_entropies = -np.sum(average_probabilities * np.log(average_probabilities), axis=1)
        return model_entropies - average_entropies