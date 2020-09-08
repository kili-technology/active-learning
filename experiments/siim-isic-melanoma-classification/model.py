import logging
import os

import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import pretrainedmodels
from wtfml.engine import Engine
from sklearn import metrics

from al.model.active_model import ActiveLearner

DATA_PATH = os.getenv('TO_DATA_PATH')


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()

        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    f"{DATA_PATH}/siic-isic-224x224-images/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss


class SEResnext50_32x4dLearner(ActiveLearner):

    def __init__(self, device=0, logger_name=None):
        super().__init__(device=device)
        self.model = SEResnext50_32x4d(pretrained="imagenet")
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.Engine = Engine(self.model, self.optimizer, device)
        self.logger = logging.getLogger(logger_name)

    def inference(self, dataset, bs=64):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False, num_workers=4
        )
        predictions = self.Engine.predict(loader)
        predictions = np.vstack((predictions)).ravel()
        probabilities = 1 / (1 + np.exp(-predictions))
        probabilities = np.stack([probabilities, 1-probabilities], axis=1)
        return {'class_probabilities': probabilities}

    def fit(self, train_dataset, epochs=50, train_bs=32, **kwargs):
        labeled_targets = [x['targets'].numpy()
                           for x in tqdm.tqdm(train_dataset)]
        train_target_distrib = pd.value_counts(labeled_targets)
        self.logger.info('Targets labeled distribution :')
        self.logger.info(train_target_distrib)

        if self.cuda_available:
            self.model.cuda()
        self.model.train()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in tqdm.tqdm(range(epochs)):
            train_loss = self.Engine.train(train_loader)
        return {'target_distribution': train_target_distrib}

    def score(self, valid_dataset, batch_size=64):
        self.model.eval()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        # valid_loss = self.Engine.evaluate(valid_loader)
        # print(f'Validation loss : {valid_loss:.3f}')
        predictions = self.Engine.predict(valid_loader)
        predictions = np.vstack((predictions)).ravel()
        valid_targets = valid_dataset.targets
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"AUC = {auc:.3f}")
        return {'auc': auc}


if __name__ == '__main__':
    model = SEResnext50_32x4d(pretrained="imagenet")
    learner = SEResnext50_32x4dLearner()
