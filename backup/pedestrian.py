import cv2
import picar
import argparse
import regional_cnn
import numpy as np
import pandas as pd
import cv2.cv as cv
from PIL import Image
from time import sleep
from imutils.video import FPS
from datetime import datetime
from datetime import datetime
from resizeimage import resizeimage
from imutils.video import VideoStream
from picar.SunFounder_PCA9685 import Servo
from picar import front_wheels, back_wheels

picar.setup()
# Show image captured by camera, True to turn on, youwill need #DISPLAY and it also slows the speed of tracking
show_image_enable   = False
draw_circle_enable  = False
scan_enable         = False
rear_wheels_enable  = True
front_wheels_enable = True
pan_tilt_enable     = True
SPEED = 60

kernel = np.ones((5,5),np.uint8)
img = cv2.VideoCapture(-1)

SCREEN_WIDTH = 160
SCREEN_HIGHT = 120
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
BALL_SIZE_MIN = SCREEN_HIGHT/10
BALL_SIZE_MAX = SCREEN_HIGHT/3

# Filter setting, DONOT CHANGE
hmn = 12
hmx = 37
smn = 96
smx = 255
vmn = 186
vmx = 255

# camera follow mode:
# 0 = step by step(slow, stable), 
# 1 = calculate the step(fast, unstable)
follow_mode = 1

CAMERA_STEP = 2
CAMERA_X_ANGLE = 20
CAMERA_Y_ANGLE = 20

MIDDLE_TOLERANT = 5
PAN_ANGLE_MAX   = 170
PAN_ANGLE_MIN   = 10
TILT_ANGLE_MAX  = 150
TILT_ANGLE_MIN  = 70
FW_ANGLE_MAX    = 90+30
FW_ANGLE_MIN    = 90-30

SCAN_POS = [[20, TILT_ANGLE_MIN], [50, TILT_ANGLE_MIN], [90, TILT_ANGLE_MIN], [130, TILT_ANGLE_MIN], [160, TILT_ANGLE_MIN], 
			[160, 80], [130, 80], [90, 80], [50, 80], [20, 80]]

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
picar.setup()

fw.offset = 0
pan_servo.offset = 10
tilt_servo.offset = 0

bw.speed = 0
fw.turn(90)
pan_servo.write(90)
tilt_servo.write(90)

motor_speed = 60

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
	# help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
	# help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
	# help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
sleep(2.0)
fps = FPS().start()

def diffImg(t0, t1, t2): # Function to calculate difference between images.
	d1 = cv2.absdiff(t2, t1)
	d2 = cv2.absdiff(t1, t0)
	return cv2.bitwise_and(d1, d2)

def nothing(x):
	pass

def main():

	pan_angle = 90  # initial angle for pan
	tilt_angle = 90 # initial angle for tilt
	fw_angle = 90
	scan_count = 0
	print "Begin!"
	
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		# blob = cv2.dnn.blobFromImage(cv2.resize(frame, (16, 16))
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		newframe = resizeimage.resize_cover(frame, [16, 16])
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()
		
		
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
		
			#####################################################
			# SEND newframe to regional_cnn class here.
			#####################################################
			# Create new r_cnn object. Initialize with image data
			r_cnn = regional_cnn.RegionalCNN(train_data=newframe, filters=32, step=32)
			pedestrian = r_cnn.train()
			print pedestrian
			if pedestrian[0][0] == 1:

				##############################################
				# Motion detection
				##############################################
				ret, frame = cam.read()	      # read from camera
				totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
				text = "threshold: " + str(totalDiff)				# make a text showing total diff.
				cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
				movement = 0
				if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):
					dimg= cam.read()[1]
					# cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
					movement = 1
				timeCheck = datetime.now().strftime('%Ss')
				# Read next image
				t_minus = t
				t = t_plus
				t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
				cv2.imshow(winName, frame)
				key = cv2.waitKey(10)
				# if key == 27:	 # comment this 'if' to hide window
				# cv2.destroyWindow(winName)
				# break
				########################################
				#CAR MOVEMENT
				########################################
				# Turn right motion detected
				if movement == 1:
					bw.speed = SPEED
					fw.turn_right()
					bw.forward()
					time.sleep(1)
					bw.stop()
					time.sleep(1)
					bw.backward()
					time.sleep(1)
					bw.stop()
					fw.turn_left()
				else:
					bw.speed = SPEED
					fw.turn_left()
					bw.forward()
					time.sleep(1)
					bw.stop()
					time.sleep(1)
					bw.backward()
					time.sleep(1)
					bw.stop()
					fw.turn_right()
			# The Prediction
			# extract the confidence (i.e., probability) associated with
			# confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			# if confidence > args["confidence"]:
				# # extract the index of the class label from the
				# # `detections`, then compute the (x, y)-coordinates of
				# # the bounding box for the object
				# idx = int(detections[0, 0, i, 1])
				# box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				# (startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				# label = "{}: {:.2f}%".format(CLASSES[idx],
					# confidence * 100)
				# cv2.rectangle(frame, (startX, startY), (endX, endY),
					# COLORS[idx], 2)
				# y = startY - 15 if startY - 15 > 15 else startY + 15
				# cv2.putText(frame, label, (startX, y),
					# cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				
		# show the output frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()