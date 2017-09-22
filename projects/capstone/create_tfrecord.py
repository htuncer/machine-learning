import numpy as np
import skimage.io as io
from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import sys
import random
from collections import namedtuple, OrderedDict
import pandas as pd
import os
PROJECT_PATH = os.getcwd()
import time


# do
# https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md


def img_to_tf_record(row, img_dir, label_map):
    """Creates a tf.Example proto from sample cat image.
    Args:
    encoded_cat_image_data: The png encoded data of the cat image.
    Returns:
    example: The created tf.Example.
    """
    img_name = row['Frame']
    img_path = img_dir + img_name
    img = np.array(Image.open(img_path))

    height = img.shape[0]
    width = img.shape[1]
    image_format = b'jpg'

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()

    xmins = [row['xmin'] / float(width)]
    xmaxs = [row['xmax'] / float(width)]
    ymins = [row['ymin'] / float(height)]
    ymaxs = [row['ymax'] / float(height)]

    label_text = [row['Label']]
    label = [label_map[row['Label']]]
    # if xmins[0] > 1.01 or xmaxs[0] > 1 or ymins[0] > 1 or ymaxs[0] > 1:
    #     print("name:%s label_text %s label %s xmin %s xmax %s ymin %s ymax %s" %
    #           (img_name, label_text[0], label[0], xmins, xmaxs, ymins, ymaxs))
    #     sys.exit(-1)

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_name),
        'image/source_id': dataset_util.bytes_feature(img_name),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(label_text),
        'image/object/class/label': dataset_util.int64_list_feature(label),
    }))

    # print("name:%s label_text %s label %s xmin %s xmax %s ymin %s ymax %s" %
    #      (img_name, label_text[0], label[0], xmins, xmaxs, ymins, ymaxs))
    return tf_record


if __name__ == "__main__":
    start = time.time()
    label_map = label_map_util.get_label_map_dict('data/label_map.pbtxt')

    label_file = PROJECT_PATH + "/data/annotated_images/labels.csv"
    img_dir = PROJECT_PATH + "/data/annotated_images/"
    data = pd.read_csv(label_file)
    #drop the rows for Pedestrian
    data = data[data.Label != 'Pedestrian']

    # in-place shuffeling
    data = data.sample(frac=1).reset_index(drop=True)

    #training_size = int(data.shape[0] * 0.7)
    training_size=10
    validation_size = int(training_size * 0.25)

    print('Creating train.record - size:%d' % training_size)
    output_path = PROJECT_PATH + "/data/train.record"
    writer = tf.python_io.TFRecordWriter(output_path)
    for index, row in data[:training_size].iterrows():
        tf_record = img_to_tf_record(row, img_dir, label_map)
        writer.write(tf_record.SerializeToString())
    writer.close()
    print ('completed train.record\n\n\n')
    print ('creating val.record - size:%d' % validation_size)
    output_path = PROJECT_PATH + "/data/val.record"
    writer = tf.python_io.TFRecordWriter(output_path)
    for index, row in data[data.shape[0]-validation_size:].iterrows():
        tf_record = img_to_tf_record(row, img_dir, label_map)
        writer.write(tf_record.SerializeToString())
    writer.close()
    print ('completed val.record\n\n\n')
    end = time.time()
    print('start:%s\nend  :%s\nDelta:%s hurs' % (start, end, (end - start) / 3600.0))
