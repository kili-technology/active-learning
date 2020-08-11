from .image_classification import *
from .object_detection import *



def set_up_learner(dataset):
    if dataset == 'mnist':
        return set_up_mnist
    elif dataset == 'cifar':
        return set_up_cifar
    elif dataset == 'pascalvoc_detection':
        return set_up_pascalvoc_detection
    elif dataset == 'pascalvoc_segmentation':
        return set_up_pascalvoc_segmentation
    elif dataset == 'coco_object_detection':
        return set_up_coco_object_detection