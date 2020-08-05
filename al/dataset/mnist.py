import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from .active_dataset import ActiveDataset
from ..helpers.constants import DATA_ROOT

class MnistDataset(ActiveDataset):

    def __init__(self, indices, n_init=100, output_dir=None, queries_name="queries.txt"):
        self.init_dataset = self._get_initial_dataset()
        super().__init__(self.get_dataset(indices), n_init=n_init, output_dir=output_dir, queries_name=queries_name)

    def _get_initial_dataset(self):
        return torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, transform=transforms.ToTensor(), download=True)

    def get_dataset(self, indices):
        return TensorDataset(
            self.init_dataset.data[indices].float() * 2.0 / 255.0 -1.0,
            self.init_dataset.targets[indices]
        )