import time

from applications.AtrousResnet50 import Atrous_Resnet50
from applications.AtrousFCN import AtrousFCN_Resnet50_16s
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.models import Model
import os
import shutil
import os.path as osp
import imageio
from PIL import Image
from skimage.io import imshow
from skimage.measure import regionprops,label
from tools.image.mar_resize import pad_resize,cut_resize
import numpy as np
import matplotlib.pyplot as plt
from segmentation.dataset.Kvasir2Keras import DATA_PATH, IMG_FMT,MASK_FMT
from keras.applications.imagenet_utils import preprocess_input

mean = [103.939, 116.779, 123.68]
mean = np.asarray(mean,dtype=np.float32)
cla_weights_PATH=r"/media/yanwe/de1dcd1c-9be8-42ed-aa06-bb73570121ac/FCN/Save_Kvasir/Classification/from_FCN_C2/weights.41.hdf5"
fcn_weights_PATH=r'/media/yanwe/de1dcd1c-9be8-42ed-aa06-bb73570121ac/FCN/Save_Kvasir/Segmentation/return_CNN_S3/weights.336.hdf5'
output_dir=osp.split(fcn_weights_PATH)[0]

if os.path.exists(osp.join(output_dir, 'fcn')):
    shutil.rmtree(osp.join(output_dir, 'fcn'))
os.makedirs(osp.join(output_dir, 'fcn'))
if os.path.exists(osp.join(output_dir, 'refine')):
    shutil.rmtree(osp.join(output_dir, 'refine'))
os.makedirs(osp.join(output_dir, 'refine'))



# fcn_weights_PATH=r"/mnt/746A723A6A71F966/Project/FCN/Save_CVC/Exps/Exp2/Segmentation/From_ClaImagenet_Adam_1e-4_amsgrad_increase91_reload/all_unlock/bestweights.302.hdf5"

# DATA_PATH= r"/mnt/746A723A6A71F966/Data/PolypDataset/CVC/KerasData/test"
# IMG_FMT='.bmp'
# DATA_PATH=r"D:\Data\PolypDataset\CVC\KerasData\test"
# IMG_FMT='.png'
# mean = [58.43420676, 87.77061877, 134.22484452]

image_path=os.path.join(DATA_PATH,'Segmentation','test','image','0')
mask_path=os.path.join(DATA_PATH,'Segmentation','test','mask','0')
nb_classes=2

FCN_threshold = 0.2
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

classes=2
input_shape=(320,320,3)
fcn_model=AtrousFCN_Resnet50_16s(input_shape=input_shape,batch_momentum=0.99,classes=classes,weights=fcn_weights_PATH)
fcn_model.summary()
base_model = Atrous_Resnet50(include_top=False, input_shape=input_shape, classes=classes)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(2, activation='softmax'))
cla_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
cla_model.load_weights(cla_weights_PATH)
cla_model.summary()
image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.endswith(IMG_FMT)]
mask_files=[f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f)) and f.endswith(MASK_FMT)]
GTmasks=[]
result_masks=[]
images=[]
GTnames=[]
time_run=[]
# c_std = np.array([36.80533591, 49.3610566, 65.13309767])
# e_std = np.array([56.30460445, 60.36653809, 66.76518424])
# std=e_std/c_std
for image_file in image_files:
    if osp.splitext(image_file)[0]+MASK_FMT in mask_files:
        image=imageio.imread(os.path.join(image_path,image_file))
        GTmask= imageio.imread(os.path.join(mask_path,osp.splitext(image_file)[0]+MASK_FMT))
        stime=time.time()
        image_resized = pad_resize(image,(320,320)).astype(np.float32)
        image_resized = image_resized[..., ::-1]
        image_resized[..., 0] -= mean[0]
        image_resized[..., 1] -= mean[1]
        image_resized[..., 2] -= mean[2]
        processed_image = np.expand_dims(image_resized, axis=0)
        result = fcn_model.predict(processed_image, batch_size=1)
        processed_result = np.squeeze(result)
        processed_result = softmax(processed_result, axis=2)
        processed_result = np.delete(processed_result, [0], axis=2)
        processed_resultimg = (processed_result * 255).astype(np.uint8)
        out_np = cut_resize(processed_resultimg, GTmask.shape[1::-1])
        time_run.append(time.time()-stime)
        GTmasks.append(GTmask)
        GTnames.append(image_file)
        images.append(image)
        out_np[out_np<255 * FCN_threshold] = 0
        out_np[out_np !=0] = 1
        if out_np.shape!= image.shape[:2]:
            raise Exception()
        im_plusmsk=image.copy()
        im_plusmsk[:,:,2][out_np==1]=im_plusmsk[:,:,2][out_np==1]*0.2+out_np[out_np==1]*0.8*255
        im_plusmsk = Image.fromarray(im_plusmsk.astype(np.uint8))
        im_plusmsk.save(osp.join(output_dir, 'fcn', osp.splitext(image_file)[0] + '.pred' + '.png'))
        im_overlap=np.zeros(image.shape)
        im_overlap[:, :, 2][out_np == 1] = out_np[out_np == 1]* 255
        im_overlap[:, :, 1][GTmask == 1] = GTmask[GTmask == 1]* 255
        im_overlap=Image.fromarray(im_overlap.astype(np.uint8))
        im_overlap.save(osp.join(output_dir, 'fcn', osp.splitext(image_file)[0]+'.overlap' + '.png'))

        im_msk=Image.fromarray(out_np)
        im_msk.putpalette([0, 0, 0,  # Background - Black
                         255, 255, 255])  # Class 1 - White
        im_msk.save(osp.join(output_dir,'fcn',osp.splitext(image_file)[0]+'.png'))
        result_masks.append(out_np)
    else:
        raise Exception()
print(np.average(time_run))
# i=0
# for (image,mask) in zip(images,result_masks):
#     f = plt.figure()
#     f.add_subplot(1, 2, 1)
#     plt.imshow(image)
#     f.add_subplot(1, 2, 2)
#     f.suptitle(image_files[i])
#     plt.imshow(mask)
#     plt.show(block=True)
#     i+=1


IOUs = np.zeros((len(GTmasks),nb_classes))
pix_accs = np.zeros((len(GTmasks),1))
for i in range(0,len(GTmasks)):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    flat_pred = np.ravel(result_masks[i])
    flat_label = np.ravel(GTmasks[i])
    IOU=np.zeros(nb_classes)
    total_intersection=0
    for p_class in range(0,nb_classes):
        p_pred=np.zeros(flat_pred.shape)
        p_pred[flat_pred==p_class]=1
        p_label=np.zeros(flat_label.shape)
        p_label[flat_label==p_class]=1
        intersection = np.logical_and(p_label, p_pred)
        union = np.logical_or(p_label, p_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        IOU[p_class]=iou_score
        total_intersection=total_intersection+np.sum(intersection)

    pix_acc=total_intersection/flat_label.size
    IOUs[i]=IOU
    pix_accs[i]=pix_acc
meanIOU=np.mean(IOUs, axis=0)
mean_pix_accs=pix_accs.mean()

    # for p, l in zip(flat_pred, flat_label):
    #     # if l == 255:
    #     #     continue
    #     if l < nb_classes and p < nb_classes:
    #         conf_m[l, p] += 1
    #     else:
    #         print('Invalid entry encountered, skipping! Label: ', l,
    #               ' Prediction: ', p, ' Img_num: ', i)

    # I = np.diag(conf_m)
    # U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    # IOU = I / U
    # meanIOU = np.mean(IOU)
    # pix_acc = np.sum(np.diag(conf_m)) / np.sum(conf_m)


# i=0
# for (image,mask) in zip(images,GTmasks):
#     f = plt.figure()
#     f.add_subplot(1, 2, 1)
#     plt.imshow(image)
#     f.add_subplot(1, 2, 2)
#     f.suptitle(image_files[i])
#     plt.imshow(mask)
#     plt.show(block=True)
#     i+=1

cla_threshold = 0.1
FCN_Total=0
FCN_TP=0
FCN_FN=0
Refine_Total=0
Refine_TP=0
Refine_FN=0
refine_masks=result_masks.copy()
Polyp_Total=0



for i in range(0, len(GTmasks)):
    stime=time.time()
    GTmask=GTmasks[i]

    image=images[i]
    if result_masks[i].max()>1 or GTmask.max()>1:
        raise Exception()

    props=regionprops(label(GTmask * 255, connectivity=2) )

    if len(props)>1:
        print("Warn: GT filename:",GTnames[i],' has ',len(props),' polyps!')
    Refine=False
    for prop in props:
        GT_bbox=prop.bbox
        FCN_Find=False
        Refine_Find=False
        Polyp_Total+=1
        for prop in regionprops( label( result_masks[i] * 255, connectivity=2) ):
            result_bbox = prop.bbox
            y0 = result_bbox[0]
            x0 = result_bbox[1]
            y1 = result_bbox[2]
            x1 = result_bbox[3]
            bbox_image = image[y0:y1, x0:x1, :]
            FCN_Total+=1
            if bb_intersection_over_union(GT_bbox,result_bbox)>=0.5:
                FCN_TP+=1
                FCN_Find=True
            image_resized = pad_resize(bbox_image, (320, 320)).astype(np.float32)
            image_resized = image_resized[..., ::-1]
            image_resized[..., 0] -= mean[0]
            image_resized[..., 1] -= mean[1]
            image_resized[..., 2] -= mean[2]
            processed_image = np.expand_dims(image_resized, axis=0)
            poss=cla_model.predict(processed_image)[0][1]
            if poss>cla_threshold:
                Refine_Total+=1
                if bb_intersection_over_union(GT_bbox, result_bbox) >= 0.5:
                    Refine_TP += 1
                    Refine_Find = True
            else:
                result_masks[i][y0:y1, x0:x1]=result_masks[i][y0:y1, x0:x1]-prop.image_filled
                Refine=True



        if not FCN_Find:
            FCN_FN+=1
        if not Refine_Find:
            Refine_FN+=1
    if Refine:
        im_plusmsk=image.copy()
        im_plusmsk[:,:,2][result_masks[i]==1]=im_plusmsk[:,:,2][result_masks[i]==1]*0.2+result_masks[i][result_masks[i]==1]*0.8*255
        im_plusmsk = Image.fromarray(im_plusmsk.astype(np.uint8))
        im_plusmsk.save(osp.join(output_dir, 'refine', osp.splitext(GTnames[i])[0] + '.pred' + '.png'))
        im_overlap=np.zeros(image.shape)
        im_overlap[:, :, 2][result_masks[i] == 1] = result_masks[i][result_masks[i] == 1]* 255
        im_overlap[:, :, 1][GTmasks[i] == 1] = GTmasks[i][GTmasks[i] == 1]* 255
        im_overlap=Image.fromarray(im_overlap.astype(np.uint8))
        im_overlap.save(osp.join(output_dir, 'refine', osp.splitext(GTnames[i])[0]+'.overlap' + '.png'))
        im_msk = Image.fromarray(result_masks[i])
        im_msk.putpalette([0, 0, 0,  # Background - Black
                           255, 255, 255])  # Class 1 - White
        im_msk.save(osp.join(output_dir, 'refine', osp.splitext(GTnames[i])[0] + '.png'))

    time_run[i]=time_run[i]+time.time()-stime
print(np.average(time_run))
FCN_FP=FCN_Total-FCN_TP
Refine_FP = Refine_Total - Refine_TP

Prec_FCN=FCN_TP/(FCN_TP+FCN_FP)
Prec_Refine=Refine_TP/(Refine_TP+Refine_FP)
Rec_FCN=FCN_TP/(FCN_TP+FCN_FN)
Rec_Refine=Refine_TP/(Refine_TP+Refine_FN)
f1=2*(Prec_Refine*Rec_Refine/(Prec_Refine+Rec_Refine))
f2=5*(Prec_Refine*Rec_Refine/(4*Prec_Refine+Rec_Refine))





IOUs_f = np.zeros((len(GTmasks),nb_classes))
pix_accs_f = np.zeros((len(GTmasks),1))
for i in range(0,len(GTmasks)):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    flat_pred = np.ravel(result_masks[i])
    flat_label = np.ravel(GTmasks[i])
    IOU_f=np.zeros(nb_classes)
    total_intersection=0
    for p_class in range(0,nb_classes):
        p_pred=np.zeros(flat_pred.shape)
        p_pred[flat_pred==p_class]=1
        p_label=np.zeros(flat_label.shape)
        p_label[flat_label==p_class]=1
        intersection = np.logical_and(p_label, p_pred)
        union = np.logical_or(p_label, p_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        IOU_f[p_class]=iou_score
        total_intersection=total_intersection+np.sum(intersection)

    pix_acc=total_intersection/flat_label.size
    IOUs_f[i]=IOU_f
    pix_accs_f[i]=pix_acc
meanIOU_f=np.mean(IOUs_f, axis=0)
mean_pix_accs_f=pix_accs_f.mean()

print("Segmentation network")
print("IoU:", meanIOU)
print("Pixel Accuracy", mean_pix_accs)
print("Segmentation network with refinement")
print("IoU:", meanIOU_f)
print("Pixel Accuracy:", mean_pix_accs_f)



















