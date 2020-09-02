from .random import *
from .uncertainty import *
from .kcenter_greedy import *
from .al_for_deep_object_detection import DeepObjectDetectionStrategy
from .deep_bayesian import *
from .diverse_mini_batch import *


def get_strategy(strategy_name, **kwargs):
    if strategy_name == 'random_sampling':
        return RandomStrategy()
    elif strategy_name == 'uncertainty_sampling':
        return UncertaintyStrategy()
    elif strategy_name == 'margin_sampling':
        return MarginStrategy()
    elif strategy_name == 'entropy_sampling':
        return EntropyStrategy()
    elif strategy_name == 'coreset':
        return KCenterGreedyStrategy(**kwargs)
    elif strategy_name == 'al_for_deep_object_detection':
        return DeepObjectDetectionStrategy(**kwargs)
    elif strategy_name == 'bayesian_entropy_sampling':
        return BayesianEntropyStrategy(**kwargs)
    elif strategy_name == 'bayesian_bald_sampling':
        return BayesianBALDStrategy(**kwargs)
    elif strategy_name == 'semantic_entropy_sampling':
        return SemanticEntropyStrategy(**kwargs)
    elif strategy_name == 'diverse_mini_batch_sampling':
        return DiverseMiniBatchStrategy(**kwargs)
