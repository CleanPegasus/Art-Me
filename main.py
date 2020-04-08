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

ROOT_DIR = os.path.abspath("../")


sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, "coco/"))
import coco

COCO_MODEL_PATH = "mask_rcnn_coco.h5"



