

class Strategy():

    def __init__(self):
        pass

    def score_dataset(self, dataset):
        raise NotImplementedError

    def return_top_indices(self, dataset, top):
        raise NotImplementedError