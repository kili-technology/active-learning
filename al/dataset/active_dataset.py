import os

import numpy as np
from torch.utils.data import Dataset

from ..helpers.query_saver import save_to_csv


class MaskDataset(Dataset):

    def __init__(self, dataset, init_indices=[]):
        super().__init__()
        self.dataset = dataset
        self.indices = init_indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class ActiveDataset():

    def __init__(self, dataset, n_init=100, output_dir=None, queries_name='queries.txt'):
        self.dataset = dataset
        self.masklabeled = np.array([False for i in range(len(dataset))])
        self.update_labeled_list()
        init_list = list(np.random.permutation(np.arange(len(dataset)))[:n_init])
        self.output_dir = output_dir
        self.queries_file = os.path.join(output_dir, queries_name)
        if os.path.exists(self.queries_file):
            os.remove(self.queries_file)
        self.add_to_labeled(init_list)

    def _get_initial_dataset(self):
        raise NotImplementedError

    def get_dataset(self, indices):
        raise NotImplementedError

    def update_labeled_list(self):
        self.labeled = [i for i, labeled in enumerate(self.masklabeled) if labeled]
        self.unlabeled_to_all = {}
        j = 0
        for i, labeled in enumerate(self.masklabeled):
            if not labeled:
                self.unlabeled_to_all[j] = i
                j += 1

    def get_labeled(self):
        return MaskDataset(self.dataset, self.labeled)

    def get_unlabeled(self):
        unlabeled_indices = [i for i, labeled in enumerate(self.masklabeled) if not labeled]
        return MaskDataset(self.dataset, unlabeled_indices)

    def add_to_labeled(self, indices):
        save_to_csv(self.queries_file, indices)
        self.masklabeled[np.array(indices)] = True
        self.update_labeled_list()

    def set_validation_dataset(self, dataset):
        self.val_dataset = dataset

    def get_validation_dataset(self):
        return self.val_dataset
