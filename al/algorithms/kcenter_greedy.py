import numpy as np
from sklearn.metrics import pairwise_distances

from .baseline import Strategy
from ..helpers.time import timeit


class KCenterGreedyStrategy(Strategy):

    def __init__(self, metric='euclidean', **kwargs):
        super().__init__()
        self.metric = metric

    @timeit
    def score_dataset(self, dataset, log_time={}):
        return None

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
        model: model with scikit-like API with decision_function implemented
        already_selected: index of datapoints already selected
        N: batch size
        Returns:
        indices of points selected to minimize distance to cluster centers
        """

        self.update_distances(
            already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch

    @timeit
    def return_top_indices(self, dataset, learner, top, log_time={}):
        self.score_dataset(dataset, log_time=log_time)
        self.min_distances = None
        self.n_obs = len(dataset)
        self.already_selected = []
        inference_result = learner.inference(dataset)
        self.features = inference_result['predictions']
        return self.select_batch_([], top)
