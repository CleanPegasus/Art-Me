import os
import sys
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

ROOT_DIR = os.path.abspath("/")

sys.path.append(ROOT_DIR)

MODEL_DIR = "logs"

from coco import CocoConfig

COCO_MODEL_PATH = "mask_rcnn_coco.h5"

class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def mask(IMAGE_PATH, BACKGROUND_PATH):

    image = skimage.io.imread(IMAGE_PATH)
    background = skimage.io.imread(BACKGROUND_PATH)

    results = model.detect([image], verbose = 1)

    mask_image = results[0]['masks']
    result_image = image*mask_image
    result_image = image*mask_image
    background = background[:result_image.shape[0], :result_image.shape[1]]
    masked_background = background * np.bitwise_not(mask_image)
    mask_result = masked_background + result_image

    return mask_result