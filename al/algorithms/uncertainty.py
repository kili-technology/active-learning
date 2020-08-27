import numpy as np

from .baseline import Strategy
from ..helpers.time import timeit

class BaseUncertaintyStrategy(Strategy):

    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        raise NotImplementedError

    def return_top_indices(self, dataset, learner, top, log_time={}):
        scores = self.score_dataset(dataset, learner, log_time=log_time)
        sorted_idx = np.argsort(scores)
        return sorted_idx[-top:]


class UncertaintyStrategy(BaseUncertaintyStrategy):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        probabilities = inference_result['class_probabilities']
        assert len(probabilities) == len(dataset)
        top_prediction = np.max(probabilities, axis=1)
        return 1 - top_prediction


class MarginStrategy(BaseUncertaintyStrategy):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        probabilities = inference_result['class_probabilities']
        assert len(probabilities) == len(dataset)
        sorted_preds = np.argsort(probabilities, axis=1)
        top_preds = probabilities[np.arange(len(probabilities)), sorted_preds[:, -1]]
        second_preds = probabilities[np.arange(len(probabilities)), sorted_preds[:, -2]]
        difference = top_preds - second_preds
        return - difference


class EntropyStrategy(BaseUncertaintyStrategy):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        probabilities = inference_result['class_probabilities']
        assert len(probabilities) == len(dataset)
        entropies = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropies

class SemanticEntropyStrategy(BaseUncertaintyStrategy):
    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        probabilities = inference_result['class_probabilities']
        bs, _, _, _ = probabilities.shape
        probabilities = np.reshape(probabilities, (bs, -1))
        entropies = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropies