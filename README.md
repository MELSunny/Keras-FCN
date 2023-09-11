# Keras FCN for polyp segmentation
This project is the implementation of the paper [Rethinking the transfer learning for FCN based polyp segmentation in colonoscopy](https://tobeconfirmed)
## Installation
```bash
conda create -n tfv1 python==3.8.13
conda activate tfv1
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod] keras==2.1.6
pip install opencv-python sklearn pillow imageio scikit-image matplotlib
```
## Usage
### Preparation

Download the dataset: 

CVC-EndoSceneStill(http://pages.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/)

[kvasir-seg.zip](https://datasets.simula.no/kvasir-seg/)

Extract the dataset compressed file to DATA_SOURCE

Define the environment path in [CVC2Keras.py](segmentation/dataset/CVC2Keras.py) and [Kvasir2Keras.py](segmentation/dataset/Kvasir2Keras.py)

* `DATA_SOURCE`: Directory for decompressed dataset

* `DATA_PATH`: Directory for converted data

* `SAVE_ROOT`: Model saving directory

### Convert the dataset

```
python segmentation/dataset/CVC2Keras.py
python segmentation/dataset/Kvasir2Keras.py
python classification/PatchGenerator.py
```
### Training the network

`./train.sh`

### Evaluate the network

Select the best model by tensorboard in evaluation dataset:
`cd` to the directory of `SAVE_ROOT`
```
tensorboard --logdir=segmentation/ --port=6006
tensorboard --logdir=classification/ --port=6006
```
Update the path of weights in [evaluate.py](detection/evaluate.py)

`cla_weights_PATH`: Path of the best classification network weights

`fcn_weights_PATH`: Path of the best FCN for segmentation network weights

