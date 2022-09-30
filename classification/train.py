from applications.AtrousResnet50 import Atrous_Resnet50
from keras_mod.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.models import Model
from keras import optimizers
from keras_mod.callbacks import TensorBoard
from keras_mod.callbacks import ModelCheckpoint,LearningRateScheduler
import os
import math
import os.path as osp
from keras.applications.imagenet_utils import preprocess_input
import argparse
from glob import glob
# weights_PATH='imagenet'
# log_dir=osp.join(SAVE_ROOT,'Classification','Imagenet_SGD_1e-3')
# DATA_PATH=r'E:\CVC-EndoSceneStill'
# SAVE_PATH=r'D:\Project\FCN\Save_CVC\Exp2\Classification\Return_SGD1e-3_cosinedecay_Atrus'
# weights_PATH=r"D:\Project\FCN\Save_CVC\Exp2\Segmentation\From_ClaImagenet_Adam_1e-4_amsgrad_increase91\bestweights.48.hdf5"
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Can be CVC or Kvasir")
    parser.add_argument("logdir", help="logdir name")
    parser.add_argument("--weights", help="wight file or experiment path, default imagenet",default='imagenet',required=False)
    args = parser.parse_args()
    if args.task=='CVC':
        from segmentation.dataset.CVC2Keras import DATA_PATH, SAVE_ROOT, Train_Size
    elif args.task=='Kvasir':
        from segmentation.dataset.Kvasir2Keras import DATA_PATH, SAVE_ROOT, Train_Size
    else:
        raise Exception
    log_dir = osp.join(SAVE_ROOT, 'Classification', args.logdir)
    if args.weights!='imagenet' and args.weights!=None:
        if osp.isdir(osp.join(SAVE_ROOT,args.weights)):
            weights_files=glob(osp.join(SAVE_ROOT,args.weights,'weights.*.hdf5'))
            epochs_list=[int(osp.split(weights_file)[1].split('.')[1]) for weights_file in weights_files]
            epochs_list.sort()
            best_epoch=epochs_list[-1]
            args.weights=osp.join(SAVE_ROOT,args.weights,'weights.'+str(best_epoch)+'.hdf5')
            print('Load best weights of '+ 'weights.'+str(best_epoch)+'.hdf5')
    classes = 2
    input_shape = (320, 320, 3)
    batch_size = 24
    epochs = 250
    learning_rate = 1e-3
    decay_steps = 230
    alpha = 0.1

    base_model= Atrous_Resnet50(include_top=False,input_shape=input_shape,weights=args.weights,classes=classes)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(2, activation='softmax'))
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers:
        layer.trainable = True


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate,momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rotation_range=180,
                                       zoom_range=[0.8, 1.25],
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='constant',
                                       cval=0,keep_rate_padding=True,
                                       preprocessing_function=preprocess_input,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1
                                       )
    validation_datagen = ImageDataGenerator(fill_mode='constant',
                                            cval=0,
                                            keep_rate_padding=True,
                                            preprocessing_function=preprocess_input
                                            )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_PATH, 'Classification','train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        # save_to_dir=os.path.join(SAVE_PATH,"train"),
        shuffle=True)


    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(DATA_PATH, 'Classification','val'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        # save_to_dir=os.path.join(SAVE_PATH,"valid"),
        shuffle=True)


    def lr_scheduler(epoch):
        global_step = min(epoch, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = learning_rate * decayed
        print(lr)
        return (lr)

    lrscheduler = LearningRateScheduler(lr_scheduler)
    tb_callback =TensorBoard(log_dir=log_dir,histogram_freq=0,write_graph=True, write_images=False)
    checkpoint = ModelCheckpoint(filepath=os.path.join(log_dir, 'weights.{epoch:d}.hdf5'), save_weights_only=True,period=1,save_best_only=True)

    callbacks = [tb_callback,checkpoint,lrscheduler]
    model.fit_generator(train_generator,
                        steps_per_epoch=(Train_Size * 2 + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,use_multiprocessing=True,
                        workers=12)