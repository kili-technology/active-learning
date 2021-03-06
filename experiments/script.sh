#!/bin/bash

if [ $1 == 'mnist' ]
then
    cd mnist_simple
    rm -f nohup.out
    nohup python mnist.py &
elif [ $1 == 'cifar100' ]
then
    cd cifar100_simple
    rm -f nohup.out
    nohup python cifar100.py &
elif [ $1 == 'cifar10' ]
then
    cd cifar10_simple
    rm -f nohup.out
    nohup python cifar.py &
elif [ $1 == 'pascal_object' ]
then
    cd pascal_voc_object_detection
    rm -f nohup.out
    nohup python pascal_voc.py &
elif [ $1 == 'coco' ]
then
    cd coco_simple
    rm -f nohup.out
    nohup python main.py &
elif [ $1 == 'pascal_semantic' ]
then
    cd pascal_voc_segmentation
    rm -f nohup.out
    nohup python pascal_voc.py &
elif [ $1 == 'use_case' ]
then
    cd siim-isic-melanoma-classification
    rm -f nohup.out
    nohup python main.py &
fi