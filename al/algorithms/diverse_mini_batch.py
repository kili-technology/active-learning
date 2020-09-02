"""
From Diverse Mini Batch Active Learning, https://arxiv.org/pdf/1901.05954.pdf
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .baseline import Strategy
from ..helpers.time import timeit


class DiverseMiniBatchStrategy(Strategy):

    def __init__(self, beta=50, **kwargs):
        self.beta = beta

    def get_informativeness(self, probabilities):
        sorted_preds = np.argsort(probabilities, axis=1)
        top_preds = probabilities[np.arange(
            len(probabilities)), sorted_preds[:, -1]]
        second_preds = probabilities[np.arange(
            len(probabilities)), sorted_preds[:, -2]]
        informativeness = 1 - (top_preds - second_preds)
        return informativeness

    @timeit
    def evaluate_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        probabilities = inference_result['class_probabilities']
        features = inference_result['features']
        return probabilities, features

    def perform_kmeans(self, features, informativeness, top):
        kmeans = KMeans(n_clusters=top, n_init=1, max_iter=1000)
        kmeans.fit(features, sample_weight=informativeness)
        return kmeans

    def return_top_indices(self, dataset, learner, top, log_time={}):
        probabilities, features = self.evaluate_dataset(
            dataset, learner, log_time=log_time)
        informativeness = self.get_informativeness(probabilities)

        sorted_idx = np.argsort(informativeness)
        n_preselected = int(self.beta * top)
        selected = sorted_idx[-n_preselected:]

        selected_features = features[selected]

        kmeans = self.perform_kmeans(
            selected_features, informativeness[selected], top)
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, selected_features)
        indices = selected[closest]
        return indices
