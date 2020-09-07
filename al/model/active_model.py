"""
Base class for an active learner (wraps a model)
"""

import torch


class ActiveLearner():

    def __init__(self, device=0):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.cuda.set_device(device)

    def fit(self, dataset):
        """
        Fit the model on a dataset
        """
        raise NotImplementedError

    def score(self, dataset):
        """
        Score the model on a dataset
        """
        raise NotImplementedError

    def inference(self, dataset):
        """
        Produce inference results on the dataset
        """
        raise NotImplementedError

    def query(self, dataset, algorithm, query_size):
        """
        Using a `dataset`, an active learning `algorithm` and a size of samples to query `query_size`,
        returns the results of the algorithm given the inference results.
        """
        inference_object = self.inference(dataset)
        return algorithm(inference_object, query_size)
