# C. Brust, C. Käding and J. Denzler (2019) Active learning for deep object detection.
# In VISAPP. https://arxiv.org/pdf/1809.09875.pdf
import logging
from collections import defaultdict

import torch
import numpy as np
import tqdm

from .baseline import Strategy
from ..helpers.time import timeit


class DeepObjectDetectionStrategy(Strategy):

    def __init__(self, logger_name=None, agregation='sum', weighted=False, labeled_ds=None):
        self.agregation = agregation
        self.weighted = weighted
        self.permut = True
        self.batch_size = 10
        self.logger = logging.getLogger(logger_name)
        if labeled_ds:
            raw_dataset = labeled_ds.dataset.dataset
            self.n_classes = len(raw_dataset.class_names)
            self.logger.debug(f'Dataset has size {len(labeled_ds)}, with {self.n_classes} classes')
            class_to_instance_size = defaultdict(int)
            for i in range(len(labeled_ds)):
                _, annotations = raw_dataset.get_annotation(i)
                labels = annotations[1]
                for label in labels:
                    class_to_instance_size[label] += 1
            self.logger.debug(f'Class to instance size : {class_to_instance_size}')
            self.logger.debug(f'Total instance count : {sum(list(class_to_instance_size.values()))}')
            self.class_to_instance_size = class_to_instance_size

    @timeit
    def score_dataset(self, dataset, learner, log_time={}):
        inference_result = learner.inference(dataset)
        detections, unlabeled_ids = inference_result['detections'], inference_result['image_ids']
        self.id2uncertaintyId = {i: k for k, i in enumerate(unlabeled_ids)}
        self.uncertaintyId2id = {v: k for k, v in self.id2uncertaintyId.items()}
        weights_parameters = {}
        if self.weighted:
            weights_parameters['n_classes'] = self.n_classes
            weights_parameters['class_to_instance_size'] = self.class_to_instance_size
        return compute_uncertainties(detections, self.agregation, self.weighted, weights_parameters)

    @timeit
    def return_top_indices(self, dataset, learner, top, log_time={}):
        uncertainties = self.score_dataset(dataset, learner, log_time=log_time)
        query_uncertainty_idx = select_top_indices(
            uncertainties, permut=self.permut, batch_size=self.batch_size, n_instances=top)
        query_uncertainty_idx = list(map(int, query_uncertainty_idx))
        assert max(query_uncertainty_idx) < len(uncertainties)
        return query_uncertainty_idx


def compute_uncertainties_asset(detection, weighted=False, weights_parameters={}):
    probas = detection['logits'].softmax(axis=1)
    labels = detection['labels']
    probas = probas.detach().cpu().numpy()
    rev = np.sort(probas, axis=1)[:, ::-1]
    v1_vs_2 = (1 - rev[:, 0] - rev[:, 1])**2
    if weighted:
        predicted_class = int(np.argsort(probas, axis=1)[:, ::-1][0, 0])
        c2isize = weights_parameters['class_to_instance_size']
        n_instances = sum(list(c2isize.values()))
        weight = (n_instances + weights_parameters['n_classes']) / (1 + c2isize[predicted_class])
    else: 
        weight = 1
    return weight * v1_vs_2, labels

def agregate_detection_uncertainties(agregation_method):
    def agregation(values, labels):
        if agregation_method == 'mean':
            return values.mean()
        elif agregation_method == 'sum':
            return values.sum()
        elif agregation_method == 'max':
            return values.max()
    return agregation

def compute_uncertainties(detections, agregation, weighted, weights_parameters):
    agregate_func = agregate_detection_uncertainties(agregation)
    uncertainties = np.array(list(map(
        lambda x: agregate_func(
            *compute_uncertainties_asset(x, weighted, weights_parameters)), detections)))
    return uncertainties


def select_top_batches(uncertainties, n_instances=100, batch_size=10):    
    n_samples = len(uncertainties)
    batch_uncertainties = np.array([
        uncertainties[i:i+batch_size].sum()
        for i in range(0, n_samples, batch_size)])
    last_batch_size = n_samples % batch_size
    if last_batch_size > 0:
        batch_uncertainties[-1] *= batch_size / last_batch_size
    ranked = np.argsort(batch_uncertainties)[::-1]
    n_batches = int(np.floor(n_instances / batch_size))
    selected_indices = []
    n_selected = 0
    for batch_id in ranked[:n_batches-1]:
        selected_indices += [s for s in list(range(batch_size*batch_id, batch_size*(batch_id+1))) if s<n_samples]
        n_selected += batch_size
    if batch_size < n_instances:
        batch_id = ranked[n_batches-1]
        selected_indices += [s for s in list(np.random.choice(
            list(range(batch_size*batch_id, batch_size*(batch_id+1))),
            size=n_instances-n_selected, replace=False)) if s<n_samples]
    return selected_indices

def select_top_indices(uncertainties, permut=True, n_instances=100, batch_size=10):
    if permut:
        permutation = np.random.permutation(len(uncertainties))
        inverse_permutation = np.argsort(permutation)
        permuted2original = dict(zip(inverse_permutation, np.arange(len(permutation))))
        permuted_uncertainties = uncertainties[permutation]
        selected_idx_permuted = select_top_batches(permuted_uncertainties,
                n_instances=n_instances, batch_size=batch_size)
        return np.array([permuted2original[x] for x in selected_idx_permuted])
    else:
        return np.array(select_top_batches(uncertainties, n_instances=n_instances, batch_size=batch_size))

