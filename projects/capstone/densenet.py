# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import log_loss
from keras.datasets import cifar10
import cv2
import numpy as np
from keras.utils import np_utils
import os
import sys
import tensorflow as tf
import random
from sklearn.metrics import average_precision_score
from densenet_custom_layers import Scale
from keras.models import model_from_json
#from load_cifar10 import load_cifar10_data
import time


def yo(img_rows, img_cols):
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    nb_train_samples = 2  # 3000 training samples
    nb_valid_samples = 2  # 100 validation samples
    num_classes = 10

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1, 2, 0), (img_rows, img_cols)).transpose(
            2, 0, 1) for img in X_train[:nb_train_samples, :, :, :]])
        X_valid = np.array([cv2.resize(img.transpose(1, 2, 0), (img_rows, img_cols)).transpose(
            2, 0, 1) for img in X_valid[:nb_valid_samples, :, :, :]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows, img_cols))
                            for img in X_train[:nb_train_samples, :, :, :]])
        X_valid = np.array([cv2.resize(img, (img_rows, img_cols))
                            for img in X_valid[:nb_valid_samples, :, :, :]])

    print("\n\n image\n\nShape:%s\n\n%s\n\n" % (X_train.shape, type(X_train)))

    print('Y_train1 shape %s\nY_train[0]=> %s' % (Y_train.shape, Y_train[0]))
    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)
    print('Y_train2 shape %s\n Y_train2[0]=> %s' % (Y_train.shape, Y_train[0]))
    return X_train, Y_train, X_valid, Y_valid


def load_data1(img_rows, img_cols):
    num_classes = 2
    img1 = cv2.resize(cv2.imread('data/labeled_images/image0839.png'),
                      (img_rows, img_cols)).astype(np.float32)
    #img2 = cv2.resize(cv2.imread('data/labeled_images/vehicle/GTI_Far/image0002.png'),
    #                  (img_rows, img_cols)).astype(np.float32)
    #img3 = cv2.resize(cv2.imread('data/labeled_images/vehicle/GTI_Far/image0003.png'),
    #                  (img_rows, img_cols)).astype(np.float32)

    print("1 img size: %s"%sys.getsizeof(img1))
    sys.exit(-1)
    for x in (img1, img2, img3):
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68

    X_train = np.array([img1, img2])
    X_valid = np.array([img3])

    Y_train = np.array([[0], [0]])
    Y_valid = np.array([[0]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)

    return X_train, Y_train, X_valid, Y_valid

def load_data(img_rows, img_cols):
    num_classes = 2
    # vehicles 8792
    # non-vehicles 8968
    re_path = "%s/data/labeled_images/vehicles/*/*.png" % os.getcwd()
    vehicles = tf.gfile.Glob(re_path)  # f has list of file paths
    print("vehicles %s" % len(vehicles))
    re_path = "%s/data/labeled_images/non-vehicles/*/*.png" % os.getcwd()
    non_vehicles = tf.gfile.Glob(re_path)  # f has list of file paths
    print("non-vehicles %s" % len(non_vehicles))
    images = list()
    x = 32
    for img_path in vehicles[:x]:
        img = cv2.resize(cv2.imread(img_path), (img_rows, img_cols)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        images.append((1,img))
    del vehicles

    for img_path in non_vehicles[:x]:  # TODO Check if label should start from 0 or 1
        img = cv2.resize(cv2.imread(img_path), (img_rows, img_cols)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        images.append((0, img))
    del non_vehicles

    shuffled_index = list(range(len(images)))
    random.seed(5413210)
    random.shuffle(shuffled_index)
    images = [images[i] for i in shuffled_index]

    del shuffled_index

    split = int(len(images) * 0.7)

    X_list = [img[1] for img in images]
    Y_list = [img[0] for img in images]
    del images

    X_train = np.array(X_list[:split])
    X_valid = np.array(X_list[split:])
    del X_list

    Y_train = np.array(Y_list[:split])
    Y_valid = np.array(Y_list[split:])

    del Y_list

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)
    print("shapes %d %d %d %d"%(X_train.shape[0], Y_train.shape[0], X_valid.shape[0], Y_valid.shape[0]))
    return X_train, Y_train, X_valid, Y_valid



def densenet161_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=1000, weights_path=None):
    '''
    DenseNet 161 Model for Keras
    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[
                                   block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(
        x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' +
                           str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    # HASAN 1000 to num_classes
    if weights_path is None:
        x_fc = Dense(1000, name='fc6')(x_fc)
    else:
        x_fc = Dense(num_classes, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if weights_path is None and K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'data/densenet161_weights_th.h5'
    elif weights_path is None:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'data/densenet161_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

if __name__ == '__main__':

    start_time = time.time()
    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 2
    batch_size = 1
    nb_epoch = 10

    print('\n\n\n\ngetting data')
    X_train, Y_train, X_valid, Y_valid = load_data(img_rows, img_cols)

    # Load our model
    print('\n\n\n\nCreating model')
    model = densenet161_model(img_rows=img_rows, img_cols=img_cols,
                              color_type=channel, num_classes=num_classes)

    # Start Fine-tuning
    print('\n\n\n\ntraining model')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=0,
              validation_data=(X_valid, Y_valid),
              )

    print('\n\n\n\nSaving Model')
    model_file = 'data/DenseNet_finetuned_std.h5'
    model.save_weights(model_file)

    print('\n\n\n\nLoading Model')
    finetuned_model = densenet161_model(img_rows=img_rows, img_cols=img_cols,
                                        color_type=channel, num_classes=num_classes, weights_path=model_file)

    print('\n\n\n\nPredicting')
    predictions_valid = finetuned_model.predict(X_valid, batch_size=batch_size, verbose=1)
    print("predictions valid %s" % predictions_valid)
    # average precision
    score = average_precision_score(Y_valid, predictions_valid)
    print("score %s" % score)
    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print('Duration: %s' % duration)
