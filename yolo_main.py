import cv2
import matplotlib.pyplot as plt

from utils import *
from darknet import Darknet

# Setting up the Neural Network
# =============================

# Set the location and name of the cfg file
cfg_file = './cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/yolov3.weights'

# Set the location and name of the COCO (common objects in context) object classes file
namesfile = 'data/coco.names'

# Load the network architecture
yolo_darknet = Darknet(cfg_file)

# Load the pre-trained weights
yolo_darknet.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

# Print the neural network architecture
yolo_darknet.print_network()

# Load and pre-process an image
# =============================

# Set the default figure size
plt.rcParams['figure.figsize'] = [24.0, 14.0]

# Load the image
img = cv2.imread('images/dog.jpg')

# Convert the image to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# We rescale the image to 416x416 to be compatible with the first layer of the darknet model
resized_image = cv2.resize(original_image, (yolo_darknet.width, yolo_darknet.height))

# Display the image
# plt.subplot(121) # 121 means 1x2 grid, first subplot
# plt.title('Original Image')
# plt.imshow(original_image)
# # plt.axis('off')
# plt.subplot(122) # 122 means 1x2 grid, second subplot
# plt.title('Resized Image')
# plt.imshow(resized_image)
# plt.show()

# Fine-Tuning YOLOv3
# ========================

# Set the NMS threshold
# YOLO uses Non-Maximal Suppression (NMS) to only keep the best bounding box. The first step
# in NMS is to remove all the predicted bounding boxes that have a detection probability that
# is less than a given NMS threshold. In the code below, we set this NMS threshold to 0.6.
# This means that all predicted bounding boxes that have a detection probability less than
# 0.6 will be removed
nms_thresh = 0.6

# Set the IOU threshold
# The Intersection over Union (IOU) threshold is used to determine how much overlap is
# required for two bounding boxes to be considered the same object. In the code below, we
# set the IOU threshold to 0.4
iou_thresh = 0.4

# Object Detection
# ================
boxes = detect_objects(yolo_darknet, resized_image, iou_thresh, nms_thresh)

print_objects(boxes, class_names)

# Display the image with bounding boxes
plot_boxes(original_image, boxes, class_names, plot_labels=True)
