#!/bin/bash
export PYTHONPATH=$PWD

python segmentation/train.py CVC imagenet_S1 --weights imagenet
python classification/train.py CVC from_FCN_C2 --weights Segmentation/imagenet_S1
python segmentation/train.py CVC return_CNN_S3 --weights Classification/from_FCN_C2
python classification/train.py CVC imagenet_C1 --weights imagenet
python segmentation/train.py CVC from_CNN_S2 --weights Classification/imagenet_C1
python classification/train.py CVC return_FCN_C3 --weights Segmentation/from_CNN_S2
python segmentation/train.py CVC RND_init_S1

python segmentation/train.py Kvasir imagenet_S1 --weights imagenet
python classification/train.py Kvasir from_FCN_C2 --weights Segmentation/imagenet_S1
python segmentation/train.py Kvasir return_CNN_S3 --weights Classification/from_FCN_C2
python classification/train.py Kvasir imagenet_C1 --weights imagenet
python segmentation/train.py Kvasir from_CNN_S2 --weights Classification/imagenet_C1
python classification/train.py Kvasir return_FCN_C3 --weights Segmentation/from_CNN_S2

python classification/train.py Kvasir circulation_C4 --weights Segmentation/return_CNN_S3
python segmentation/train.py Kvasir circulation_S4 --weights Classification/circulation_C4

python segmentation/train.py Kvasir RND_init_S1