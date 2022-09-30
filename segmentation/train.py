import os
import os.path as osp
from keras.optimizers import Adam
from applications.AtrousFCN import AtrousFCN_Resnet50_16s

from utils.loss_function import *
from keras.applications.imagenet_utils import preprocess_input
from utils.metrics import *
from keras.optimizers import SGD,RMSprop
from keras_mod.preprocessing.image import ImageDataGenerator
from keras_mod.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler
import keras.preprocessing.image
import math
import argparse
from glob import glob
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# weights_PATH=r'/mnt/746A723A6A71F966/Project/FCN/Save_Kvasir/Classification/From_IMGNET_FCN_SGD_1e-3/weights.88.hdf5'
# weights_PATH=r'/mnt/746A723A6A71F966/Project/FCN/Save_Kvasir/Classification/Imagenet_SGD_1e-3/weights.64.hdf5'
# CVC
# DATA_PATH= r"D:\Data\PolypDataset\CVC\KerasData"
# log_dir= os.path.join("D:\\", "Project", "FCN", "Save_CVC","Exps","Exp2","Segmentation","From_ClaImagenet_Adam_1e-4_amsgrad_increase91_reload","all_unlock")
# weights_PATH=r"D:\Project\FCN\Save_CVC\Exps\Exp2\Classification\Imagenet_SGD1e-3_cosinedecay_Atrous\bestweights.92.hdf5"
# weights_PATH=r"D:\Project\FCN\Save_CVC\Exps\Exp2\Segmentation\From_ClaImagenet_Adam_1e-4_amsgrad_increase91_reload/weights.12.hdf5"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Can be CVC or Kvasir")
    parser.add_argument("logdir", help="logdir name")
    parser.add_argument("--weights", help="wight file or experiment path, default imagenet",default=None,required=False)
    args = parser.parse_args()
    if args.task=='CVC':
        from dataset.CVC2Keras import DATA_PATH, SAVE_ROOT,Train_Size,Val_Size
    elif args.task=='Kvasir':
        from dataset.Kvasir2Keras import DATA_PATH, SAVE_ROOT,Train_Size,Val_Size
    else:
        raise Exception
    log_dir = osp.join(SAVE_ROOT, 'Segmentation', args.logdir)
    if args.weights!='imagenet' and args.weights!=None:
        if osp.isdir(osp.join(SAVE_ROOT,args.weights)):
            weights_files=glob(osp.join(SAVE_ROOT,args.weights,'weights.*.hdf5'))
            epochs_list=[int(osp.split(weights_file)[1].split('.')[1]) for weights_file in weights_files]
            epochs_list.sort()
            best_epoch=epochs_list[-1]
            args.weights=osp.join(SAVE_ROOT,args.weights,'weights.'+str(best_epoch)+'.hdf5')
            print('Load best weights of '+ 'weights.'+str(best_epoch)+'.hdf5')

    classes=2
    input_shape=(320,320,3)
    model = AtrousFCN_Resnet50_16s(input_shape=input_shape, batch_momentum=0.99, classes=classes,weights=args.weights)
    batch_size = 24
    epochs=500
    if args.weights==None:
        learning_rate = 1e-3
    else:
        learning_rate=1e-4
    decay_steps=460
    alpha=0.1
    miou_metric = MeanIoU(2)
    finetune=False

    if finetune:
        for layer in model.layers[:174]:
            layer.trainable = False

    model.compile(loss=[softmax_sparse_crossentropy],
                  optimizer=Adam(lr=learning_rate,amsgrad=True),
                  metrics=[miou_metric.sparse_mean_iou])

    train_data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=180.,
                         zoom_range=[0.8, 1.25],
                         fill_mode='constant',
                         cval=0,
                         horizontal_flip=True,
                         vertical_flip=True,
                         keep_rate_padding=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1)
    train_image_datagen = ImageDataGenerator(**train_data_gen_args,preprocessing_function=preprocess_input)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    train_image_generator = train_image_datagen.flow_from_directory(
        os.path.join(DATA_PATH,'Segmentation','train','image'),
        target_size=(320, 320),
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = os.path.join(log_dir, "image"),
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        os.path.join(DATA_PATH,'Segmentation','train','mask'),
        color_mode='grayscale',
        target_size=(320, 320),
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir=os.path.join(log_dir, "mask"),
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(train_image_generator, train_mask_generator)

    val_data_gen_args = dict(
        # horizontal_flip=True,
        # vertical_flip=True,
        fill_mode='constant',
        cval=0,
        keep_rate_padding=True
    )

    val_image_datagen = ImageDataGenerator(**val_data_gen_args,preprocessing_function=preprocess_input)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_args)

    val_image_generator = val_image_datagen.flow_from_directory(
        os.path.join(DATA_PATH,'Segmentation','val','image'),
        target_size=(320, 320),
        class_mode=None,
        batch_size=Val_Size,
        shuffle=False,
        seed=seed)

    val_mask_generator = val_mask_datagen.flow_from_directory(
        os.path.join(DATA_PATH,'Segmentation','val','mask'),
        color_mode='grayscale',
        target_size=(320, 320),
        class_mode=None,
        shuffle=False,
        batch_size=Val_Size,
        seed=seed)

    # combine generators into one which yields image and masks
    val_generator = zip(val_image_generator, val_mask_generator)
    val_data=next(val_generator)

    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

    checkpoint = ModelCheckpoint(filepath=os.path.join(log_dir, 'weights.{epoch:d}.hdf5'), save_weights_only=True,
                                 period=1,save_best_only=True,monitor='val_sparse_mean_iou',mode='max')  # .{epoch:d}

    def lr_scheduler(epoch):
        global_step = min(epoch, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr=learning_rate * decayed
        print(lr)
        return (lr)


    lrscheduler = LearningRateScheduler(lr_scheduler)


    callbacks = [tb_callback,checkpoint,lrscheduler]
    model.fit_generator(
        train_generator,
        steps_per_epoch=(Train_Size + batch_size - 1) // batch_size,
        validation_data=val_data,
        # validation_steps=15,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=6)