import os.path as osp
import os, shutil
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
DATA_PATH= r'/home/lincoln/Documents/Data/CVC-EndoSceneStill'
DATA_FMT='.bmp'
IMG_FMT=DATA_FMT
MASK_FMT='.png'
SAVE_ROOT='/mnt/746A723A6A71F966/Project/FCN/Save_CVC'
DATA_SOURCE='/mnt/746A723A6A71F966/Data/PolypDataset/CVC-EndoSceneStill'

Train_Size=547
Val_Size=183
Test_Size=182
mean=[58.43420676,87.77061877,134.22484452]


def list_range(a, b):
    num_list = list(range(a, b + 1))
    str_list = [str(x) for x in num_list]
    return str_list


class DataSplit:
    train = {
        'CVC-300': list_range(1, 76) + list_range(98, 148) + list_range(221, 273),
        'CVC-612': list_range(26, 50) + list_range(104, 126) + list_range(178, 227) + list_range(253, 317)
                   + list_range(384, 503) + list_range(529, 612)}

    val = {
        'CVC-300': list_range(77, 97) + list_range(209, 220) + list_range(274, 300),
        'CVC-612': list_range(51, 103) + list_range(228, 252) + list_range(318, 342) + list_range(364, 383)}

    test = {
        'CVC-300': list_range(149, 208),
        'CVC-612': list_range(1, 25) + list_range(127, 177) + list_range(343, 363) + list_range(504, 528)}

def convert():
    for subset in ['train','val','test']:
        t_images_folder = osp.join(DATA_PATH, 'Segmentation', subset, 'image', '0')
        os.makedirs(t_images_folder)
        t_masks_folder = osp.join(DATA_PATH, 'Segmentation', subset, 'mask', '0')
        os.makedirs(t_masks_folder)
        for sets in getattr(DataSplit,subset):
            for item in getattr(DataSplit,subset)[sets]:
                shutil.copyfile(osp.join(DATA_SOURCE, sets,'bbdd', item + IMG_FMT), osp.join(t_images_folder, sets + '_' + item+ IMG_FMT))
                s_mask_image = Image.open(os.path.join(DATA_SOURCE, sets, 'gtpolyp', item + IMG_FMT)).convert('L')
                s_mask_array = np.asarray(s_mask_image).astype(np.uint8)
                s_mask_array[s_mask_array <= 127] = 0
                s_mask_array[s_mask_array > 127] = 1
                s_mask_image = Image.fromarray(s_mask_array)
                s_mask_image.save(osp.join(t_masks_folder, sets + '_' + item + MASK_FMT))


if __name__ == '__main__':
    if osp.exists(osp.join(DATA_PATH, 'Segmentation')):
        shutil.rmtree(osp.join(DATA_PATH, 'Segmentation'))
    if not osp.exists(SAVE_ROOT):
        os.makedirs(osp.join(SAVE_ROOT))
    convert()


