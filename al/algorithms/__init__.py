from .random import *
from .uncertainty import *
from .coreset import *
from .al_for_deep_object_detection import DeepObjectDetectionStrategy


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
        return CoreSetStrategy(**kwargs)
    elif strategy_name == 'al_for_deep_object_detection':
        return DeepObjectDetectionStrategy(**kwargs)