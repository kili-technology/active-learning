import os

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import *


mnist = True
cifar10 = False
cifar100 = False
pascal = False

if mnist:
    folder = os.path.join(os.path.dirname(__file__),
                          '../', 'mnist_simple', 'results')
    # produce_figures(f'{folder}/queries-*simplenet.txt')
    produce_figures(f'{folder}/queries-*simple_cnn.txt')

if cifar10:
    folder = os.path.join(os.path.dirname(__file__),
                          '../', 'cifar10_simple', 'results')
    produce_figures(f'{folder}/queries-*mobilenet.txt')
    # produce_figures(f'{folder}/queries-*nasnet.txt')


if cifar100:
    folder = os.path.join(os.path.dirname(__file__),
                          '../', 'cifar100_simple', 'results')
    # produce_figures(f'{folder}/queries-*simplenet.txt')
    produce_figures(f'{folder}/queries-*mobilenet.txt')

if pascal:
    folder = os.path.join(os.path.dirname(__file__),
                          '../', 'pascal_voc_object_detection', 'results')
    # produce_figures(f'{folder}/queries-*simplenet.txt')
    produce_figures(f'{folder}/queries-*mobilenet_v2.txt')
