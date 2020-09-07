# active-learning

This repository contains code developed for [Kili Technology](https://kili-technology.com/), to investigate the use of active learning to accelerate the training pipeline. Active learning is a used to select which samples out of an unlabeled dataset should be added to the training data to maximize your model accuracy.

## Summary

The repository is divided as follows :

- In `active-learning`, you can find a library containing models, dataset, and algorithms used for active learning.
  - In `algorithms`, different classes of algorithms are reproduced.
  - In `dataset`, academic dataset wrappers adapted to the active learning framework are defined.
  - In `experiments`, there are useful functions for setting up experiments.
  - In `helpers`, you can find things like a logger, a timer, etc...
  - In `model`, you have backbones of models used to produce the experiments in `model_zoo/` and a wrapper around those models to support active learning at the root of the folder.
  - In `train`, you have `active_train.py` which contains a class used to train a model in an active-learning fashion.
- In `experiments`, you can find the code for different experiments ran.

## Get started

```
git clone https://github.com/mDuval1/active-learning
cd active-learning
pip install .
```
