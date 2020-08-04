import numpy as np

from .baseline import Strategy
from ..helpers.time import timeit


class RandomStrategy(Strategy):

    def __init__(self):
        super().__init__()

    @timeit
    def score_dataset(self, dataset, log_time={}):
        return np.random.rand(len(dataset))

    @timeit
    def return_top_indices(self, dataset, learner, top, log_time={}):
        scores = self.score_dataset(dataset, log_time=log_time)
        sorted_idx = np.argsort(scores)
        return sorted_idx[-top:]