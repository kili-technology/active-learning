"""
Base class for an active learning scoring based strategy
"""


class Strategy():

    def __init__(self):
        pass

    def score_dataset(self, dataset, learner):
        raise NotImplementedError

    def return_top_indices(self, dataset, top):
        raise NotImplementedError
