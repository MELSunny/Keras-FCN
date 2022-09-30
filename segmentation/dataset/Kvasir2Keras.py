import os.path as osp
import os, shutil
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
DATA_PATH= r'/home/yanwe/Documents/Data/Kvasir'
DATA_FMT='.jpg'
IMG_FMT=DATA_FMT
MASK_FMT='.png'
SAVE_ROOT='/media/yanwe/de1dcd1c-9be8-42ed-aa06-bb73570121ac/FCN/Save_Kvasir'
DATA_SOURCE='/mnt/746A723A6A71F966/Data/PolypDataset/Kvasir-SEG'
Train_Size=800
Val_Size=100
mean=[60.10156333,81.99443822,142.01043286]
std=[48.08492297,56.76511403,81.34908145]

def convert():
    s_images_folder=osp.join(DATA_SOURCE,'images')
    s_masks_folder=osp.join(DATA_SOURCE,'masks')
    all_images=glob(osp.join(DATA_SOURCE,'images','*'+DATA_FMT))
    all_images=[osp.basename(image) for image in all_images]
    train_val_images, test_images = train_test_split(all_images, test_size=100/1000, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=100/900,  random_state=42)
    for subset, images in zip(['train','val','test'],[train_images,val_images,test_images]):
        t_images_folder=osp.join(DATA_PATH,'Segmentation', subset, 'image', '0')
        os.makedirs(t_images_folder)
        t_masks_folder=osp.join(DATA_PATH,'Segmentation', subset, 'mask', '0')
        os.makedirs(t_masks_folder)
        for image in images:
            shutil.copyfile(osp.join(s_images_folder,image),osp.join(t_images_folder,image))
            s_mask_image=Image.open(osp.join(s_masks_folder,image)).convert('L')
            s_mask_array=np.asarray(s_mask_image).astype(np.uint8)
            s_mask_array[s_mask_array <= 127] = 0
            s_mask_array[s_mask_array > 127] = 1
            s_mask_image=Image.fromarray(s_mask_array)
            s_mask_image.save(osp.join(t_masks_folder,osp.splitext(image)[0]+MASK_FMT))

if __name__ == '__main__':
    if osp.exists(osp.join(DATA_PATH, 'Segmentation')):
        shutil.rmtree(osp.join(DATA_PATH, 'Segmentation'))
    if not osp.exists(SAVE_ROOT):
        os.makedirs(osp.join(SAVE_ROOT))
    convert()