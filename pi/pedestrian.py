
import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import cv2.cv as cv
import picar
import regional_cnn
import pandas as pd
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from resizeimage import resizeimage
from itertools import chain

from utils import label_map_util

from utils import visualization_utils as vis_util
from picar.SunFounder_PCA9685 import Servo
from picar import front_wheels, back_wheels
from datetime import datetime

picar.setup()
# Show image captured by camera, True to turn on, youwill need #DISPLAY and it also slows the speed of tracking
show_image_enable   = False
draw_circle_enable  = False
scan_enable         = False
rear_wheels_enable  = True
front_wheels_enable = True
pan_tilt_enable     = True
SPEED = 60

# # Model preparation
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
picar.setup()

fw.offset = 0
# pan_servo.offset = 10
pan_servo.offset = 0
tilt_servo.offset = 0

bw.speed = 0
# fw.turn(90)
# pan_servo.write(90)
# tilt_servo.write(90)



# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print ('Download complete')
else:
    print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def diffImg(t0, t1, t2): # Function to calculate difference between images.
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

	
# intializing the web camera device
# import cv2
#threshold = 78000
threshold = 45000
cap = cv2.VideoCapture(0)


def main():
    # pan_angle = 90  # initial angle for pan
    # tilt_angle = 90 # initial angle for tilt
    # fw_angle = 90
    scan_count = 0
    print "Begin!"

    # Read three images first:
    t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    # Lets use a time check so we only take 1 pic per sec
    timeCheck = datetime.now().strftime('%Ss')

    count = 0
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                count += 1
                print "start iteration: %s" % count
                ret,image_np = cap.read()
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
                print "Get frame and resize 16x16"
                resizeFrame = cv2.resize(image_np, (16, 16))
                rgbFrame = [x for sets in resizeFrame for x in sets]
                newFrame = list(chain.from_iterable(rgbFrame))
                # print "length of image cols %s: " % len(newFrame)
                # print newFrame
                # Initializing R-CNN
                # print "Initializing R-CNN"
                r_cnn = regional_cnn.RegionalCNN(train_data=newFrame, filters=32, step=32)
                print "getting prediction from R-CNN"
                pedestrian = r_cnn.train()
                print "pedestrian classifier %s" % pedestrian
                movement = 0
                # print "pedestrian[0][0] = %s" % pedestrian[0][0]
                if pedestrian[0][0] == 1:
                    ##############################################
                    # Motion detection
                    ##############################################
                    print "start motion detection"
                    ret, frame = cap.read()	      # read from camera
                    totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
                    text = "threshold: " + str(totalDiff)				# make a text showing total diff.
                    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
                    # movement = 0
                    if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):
                        dimg= cap.read()[1]
                        cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
                        # pedestrian moving.
                        movement = 1
                    timeCheck = datetime.now().strftime('%Ss')
                    # # Read next image
                    t_minus = t
                    t = t_plus
                    t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
                    # # cv2.imshow(winName, frame)
                    # key = cv2.waitKey(10)
                    # if key == 27:	 # comment this 'if' to hide window
                    # cv2.destroyWindow(winName)
                    # break
                    ########################################
                    #CAR MOVEMENT
                    ########################################
                    # Turn right motion detected
                    if movement == 1:
                        print "turn right, pedestrian moving"
                        bw.speed = SPEED
                        fw.turn_right()
                        bw.forward()
                        time.sleep(1)
                        bw.stop()
                        time.sleep(1)
                        bw.speed = SPEED
                        bw.backward()
                        time.sleep(1)
                        bw.stop()
                        fw.turn_left()
                    else:
                        print "turn left, pedestrian not moving"
                        bw.speed = SPEED
                        fw.turn_left()
                        bw.forward()
                        time.sleep(1)
                        bw.stop()
                        time.sleep(1)
                        bw.speed = SPEED
                        bw.backward()
                        time.sleep(1)
                        bw.stop()
                        fw.turn_right()
                    time.sleep(3)
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                #      plt.figure(figsize=IMAGE_SIZE)
                #      plt.imshow(image_np)
                # cv2.imshow('image',cv2.resize(image_np,(1280,960)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    break


def destroy():
    bw.stop()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        destroy()
