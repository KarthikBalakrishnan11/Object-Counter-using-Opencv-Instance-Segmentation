# Object-Counter-using-Opencv-Instance-Segmentation
Object Counter using Opencv Instance Segmentation - Mask R-CNN

This project has done using OpenCV, Python, and Deep Learning. 

First we’ll discuss the difference between image classification, object detection, instance segmentation, and semantic segmentation.

**Image Classification:**

Image classification is a supervised learning problem, define a set of target classes (objects to identify in images), and train a model to recognize them using labeled example photos.

**Object detection:**

Object detection builds on image classification, but this time allows us to localize each object in an image. Using x,y bounding box co-ordinates with associated class label for each bounding box.

**Semantic segmentation:**

Semantic segmentation algorithms require us to associate every pixel in an input image with a class label. segmentation algorithms are capable of labeling every object in an image they cannot differentiate between two objects of the same class.

**Instance segmentation:**

Instance segmentation compute a pixel-wise mask for every object in the image, even if the objects are of the same class label. This algorithm not only localized each individual class (also same class) but predicted their boundaries as well.

**Mask R-CNN:**

Using Mask R-CNN you can automatically segment and construct pixel-wise masks for every object in an image.

Other object detection algorithms like YOLO, Faster R-CNNs, and Single Shot Detectors (SSDs), generate four sets of x, y coordinates which represent the bounding box of an object in an image. But box itself doesn’t tell us anything about foreground, background and instances.

So we need to use Mask R-CNN architecture for instance segmentation.

Mask R-CNN is an extension over Faster R-CNN. Faster R-CNN predicts bounding boxes and Mask R-CNN essentially adds one more branch for predicting an object mask in parallel.

Download the mask_rcnn_inception_v2_coco from here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
