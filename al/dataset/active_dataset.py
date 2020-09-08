"""
Base class for active learning supporting datasets
"""

import os

import numpy as np
from torch.utils.data import Dataset

from ..helpers.query_saver import save_to_csv


class MaskDataset(Dataset):
    """
    Filters a torch dataset with a list of indices.
    """

    def __init__(self, dataset, init_indices=[]):
        super().__init__()
        self.dataset = dataset
        self.indices = init_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class ActiveDataset():
    """
    Base class for a dataset supporting active learning.

    A child class should implement methods :
      - `_get_initial_dataset` : initializes the dataset
      - `get_dataset(indices)` : returns the initial dataset with selected indices
    """

    def __init__(self, dataset, n_init=100, output_dir=None, queries_name='queries.txt'):
        """
        Initialize the class

        Parameters
        ----------
        - dataset : torch.utils.data.Dataset
        - n_init : int, number of samples to initially add to the training data
        - output_dir : str, optional. Directory to write the queries to
        - queries_name : str, optional. File name of the queries made on the unlabeled dataset
        """
        self.dataset = dataset
        self.masklabeled = np.array([False for i in range(len(dataset))])
        self.update_labeled_list()
        init_list = list(np.random.permutation(
            np.arange(len(dataset)))[:n_init])
        self.save_queries = not output_dir is None
        if self.save_queries:
            self.output_dir = output_dir
            self.queries_file = os.path.join(output_dir, queries_name)
            if os.path.exists(self.queries_file):
                os.remove(self.queries_file)
        self.add_to_labeled(init_list)

    def _get_initial_dataset(self):
        """
        When adapting this class for your dataset, this should return the raw initial dataset.
        """
        raise NotImplementedError

    def get_dataset(self, indices):
        """
        When adapting this class for your dataset, this should return the initial dataset filtered by the indices
        """
        raise NotImplementedError

    def update_labeled_list(self):
        """
        Update the list of labeled and unlabeled data points, using self.masklabeled
        """
        self.labeled = [i for i, labeled in enumerate(
            self.masklabeled) if labeled]
        self.unlabeled_to_all = {}
        j = 0
        for i, labeled in enumerate(self.masklabeled):
            if not labeled:
                self.unlabeled_to_all[j] = i
                j += 1

    def get_labeled(self):
        """
        Return the labeled dataset
        """
        return MaskDataset(self.dataset, self.labeled)

    def get_unlabeled(self):
        """
        Return the unlabeled dataset
        """
        unlabeled_indices = [i for i, labeled in enumerate(
            self.masklabeled) if not labeled]
        return MaskDataset(self.dataset, unlabeled_indices)

    def add_to_labeled(self, indices):
        """
        Add a list of data points to the labeled dataset

        Parameters
        ----------
        - indices : List[int].
        """
        if self.save_queries:
            save_to_csv(self.queries_file, indices)
        self.masklabeled[np.array(indices)] = True
        self.update_labeled_list()

    def set_validation_dataset(self, dataset):
        """
        Set the validation dataset.
        """
        self.val_dataset = dataset

    def get_validation_dataset(self):
        """
        Return the validation dataset
        """
        return self.val_dataset
