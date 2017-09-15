import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
from keras.utils import np_utils
from keras.models import load_model
import sys
import os
import random
# We only test DenseNet-121 in this script for demo purpose
# from densenet import DenseNet
from sklearn.metrics import average_precision_score
import tensorflow as tf
# from DenseNet import densenet161, custom_layers
from DenseNet.densenet161 import DenseNet

sys.path.insert(0, "%s/DenseNet" % os.getcwd())
from DenseNet import custom_layers


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
    cnt = 0
    for img_path in vehicles:
        img = cv2.resize(cv2.imread(img_path), (img_rows, img_cols)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        image_map = {'label': 1, 'image': img}
        images.append(image_map)
        cnt += 1
        if cnt == 2:
            break

    for img_path in non_vehicles:  # TODO Check if label should start from 0 or 1
        img = cv2.resize(cv2.imread(img_path), (img_rows, img_cols)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        image_map = {'label': 0, 'image': img}
        images.append(image_map)
        cnt += 1
        if cnt == 4:
            break

    shuffled_index = list(range(len(images)))
    random.seed(5413210)
    random.shuffle(shuffled_index)
    images = [images[i] for i in shuffled_index]

    split = int(len(images) * 0.7)

    X_list = [img['image'] for img in images]
    Y_list = [img['label'] for img in images]

    X_train = np.array(X_list[:split])
    X_valid = np.array(X_list[split:])

    Y_train = np.array(Y_list[:split])
    Y_valid = np.array(Y_list[split:])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)

    return X_train, Y_train, X_valid, Y_valid

# Load images.
img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 1000
batch_size = 16
nb_epoch = 10

print('before load image')
X_train, Y_train, X_valid, Y_valid = load_data(img_rows, img_cols)
print('after load image')

weights_path = 'data/densenet161_weights_tf.h5'

# Test pretrained model
# classes=num_classes,
model = DenseNet(reduction=0.5, classes=num_classes, weights_path=weights_path)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print ('DenseNet created')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )
print('DenseNet trained')

model_file = 'data/DenseNet_finetuned.h5'
model.save_weights(model_file)

num_classes = 3
model = DenseNet(reduction=0.5, classes=num_classes, weights_path=model_file)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

print("predictions valid %s" % predictions_valid)
# average precision
score = average_precision_score(Y_valid, predictions_valid)
print("score %s" % score)

sys.exit(-1)
