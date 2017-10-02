import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from densenet import densenet_model

sys.path.append(os.path.abspath('./tf_models'))
sys.path.append(os.path.abspath('./tf_models/slim'))

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#transform_img_for_densenet
def detected_obj_box_to_densenet_input_images(image, boxes, densenet_img_rows, densenet_img_cols):
    #gets full image, and boxes that faster rcnn labeled as objects
    #returns list of individual images  (densenet_img_rows x densenet_img_cols)

    #box [y_min, x_min, y_max, x_max]  normalized values

    im_width = image.shape[1]
    im_height = image.shape[0]
    images = list()
    for box in boxes:
        #denormalize
        x_min = int(box[1]*im_width)
        x_max = int(box[3]*im_width)
        y_min = int(box[0]*im_height)
        y_max = int(box[2]*im_height)
        obj = image[y_min:y_max, x_min:x_max] #img[y: y + h, x: x + w]
        img = cv2.resize(obj, (densenet_img_rows, densenet_img_cols)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        images.append(img)

    return np.array(images)

def interpred_predictions(predictions,boxes):
    #gets densenet prediction and object box coordinates
    #returns only vehicle box coordinates, confidence score and class list
    score_list=list()
    class_list = list()
    box_list = list()
    i=0
    for pred in predictions:
        if pred[0]<pred[1]:
            #print("pred: %s is car"%i)
            score_list.append(pred[1])
            class_list.append(1)
            box_list.append(boxes[i])
        i+=1
    return (np.array(box_list),np.array(score_list),np.array(class_list))




obj_detection_graph_path = 'models/faster_rcnn/output/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
labels = "data/label_map.pbtxt"
num_classes = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(obj_detection_graph_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#READ finetuned DENSENET MODEL
densenet_img_rows, densenet_img_cols = 224, 224  # Resolution of inputs
channel = 3
classification_model_path = 'models/densenet/densenet_finetuned.h5'
classification_model = densenet_model(img_rows=densenet_img_rows, img_cols=densenet_img_cols,
                                        color_type=channel, num_classes=num_classes, weights_path=classification_model_path)


#input video
#cap = cv2.VideoCapture('data/videos/test_video.mp4')
cap = cv2.VideoCapture('data/videos/project2.mp4')
#output video
video = cv2.VideoWriter('data/out_project2.avi',-1,60,(1280,720))

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video = cv2.VideoWriter("output12.avi", fourcc, 20.0, (1280,720))

i = 0
while(cap.isOpened()):
    i +=1
    #if i >100:
    #    break
    #elif i < 100:
    #    continue
    ret, image = cap.read()
    image_np  = np.asarray(image)

    #FASTER R_CNN OBJECT DETECTION
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

                max_output_size = 5
                iou_threshold =0.5
                #,iou_threshold=iou_threshold
                #print("scores %s"%scores[0])
                selected_indices = tf.image.non_max_suppression( boxes[0], scores[0], max_output_size, iou_threshold=iou_threshold)
                selected_boxes = list()
                #print('selected indexis:%s'%selected_indices.eval())
                for j in selected_indices.eval():
                    selected_boxes.append(boxes[0][j])
                #print('frame %s'%i)

    #DENSENET  OBJECT CLASSIFICATION
    images=detected_obj_box_to_densenet_input_images(image_np, selected_boxes, densenet_img_rows, densenet_img_cols)
    predictions = classification_model.predict(images, batch_size=1, verbose=1)
    #print('predictions %s'%predictions)
    (vehicle_boxes, scores, classes) = interpred_predictions(predictions,selected_boxes)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,vehicle_boxes,classes,scores, category_index,use_normalized_coordinates=True,line_thickness=2)
    video.write(image_np)

cv2.destroyAllWindows()
video.release()
