# Basic python and ML Libraries
import os
import random
import math
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Function to visualize bounding boxes in the image - save the image in the figures folder
def plot_img_bbox(img, target, fig_path):

    # Remove old plot if it exists:
    if os.path.exists(fig_path):
        os.remove(fig_path)

    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        # specify cpu just in case.
        x, y, width, height  = box[0].cpu(), box[1].cpu(), box[2].cpu()-box[0].cpu(), box[3].cpu()-box[1].cpu()
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.savefig(fig_path, bbox_inches = "tight")
