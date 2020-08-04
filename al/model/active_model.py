import torch


class ActiveLearner():

    def __init__(self, device=0):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.cuda.set_device(device)

    def fit(self, dataloader):
        raise NotImplementedError

    def score(self, dataloader):
        raise NotImplementedError

    def inference(self, dataset):
        raise NotImplementedError

    def query(self, dataset, algorithm, query_size):
        inference_object = self.inference(dataset)
        return algorithm(inference_object, query_size)