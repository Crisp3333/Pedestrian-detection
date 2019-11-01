# Pedestrian Detection by a Robot Using R-CNN

Using the [Raspberry Pi Smart Video Car Kit](https://www.sunfounder.com/smart-video-car-kit-for-raspberry-pi-black.html) our aim is to detect pedestrians and classify them as either being stationary or moving. Object detection will be done via Regional Convolutional Neural Network (RCNN) . The training data that will be used is the Penn-Fudan Database for Pedestrian Detection and Segmentation https://www.cis.upenn.edu/~jshi/ped_html/ . This data will be trained to see when pedestrians are stationary and when they are moving, and for testing we will classify the active status of pedestrians in real time. The robot will indicate if the pedestrians are active (moving turn right, not moving turn left) or inactive (stationary do nothing).

![enter image description here](https://i.postimg.cc/GpWfWdvJ/rasp-car.jpg)

## Getting Started

You need to have some knowledge of the python programming language. You need to purchase the robot Raspberry Pi Car Kit V2.0 by SunFounder, along with a Raspbperry Pi and a micro sd/tf card.  The robot drivers and hardware controlled by Python code. The RCNN algorithm I wrote `regional_cnn.py` is not optimal , and therefore needs to be improved. Check out my [RCNN algorithm](https://github.com/Crisp3333/rcnn-algorithm) for a better implementation of RCNN. This is just for educational purposes for an introduction to machine learning and robotics.

Two algorithm are involved. Regional Convolutional Neural Network (RCNN) Tensorflow was the utility used for image segmentation implementation through its graph package, and the motion detection algorithm `motiondetect.py` code can be found [here]( [http://www.steinm.com/blog/motion-detection-webcam-python-opencv-differential-images/])


### Prerequisites
```
Raspberry Pi Car Kit V2.0 by SunFounder
Raspbperry Pi
micro sd/tf card
```

### Installing

For installation you can refer [to the pdf](https://www.sunfounder.com/learn/download/UGlDYXItU19Vc2VyX01hbnVhbC5wZGY=/dispi), you can also find instructions on SunFounders official youtube channel [in this video]([https://www.youtube.com/watch?v=ZCYaufyU3XA](https://www.youtube.com/watch?v=ZCYaufyU3XA)). 
When you have completed the setup
```
$ cd Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/server
```
and copy the files 
```
regional_cnn.py
pedestrian.py
motiondetect.py
```
to the `server` directory. When finished copying files run 
```
$ sudo python pedestrian.py
```
And that's it.

 [Live demo of the robot](https://www.youtube.com/watch?v=PIpQVAPMn0U)
