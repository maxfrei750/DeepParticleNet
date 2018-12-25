# Add the path of Mask_RCNN to use it as a submodule.
import os
import sys

MRCNN_DIR = os.path.abspath(os.path.join(__path__[0], "..", "external", "Mask_RCNN"))
sys.path.append(MRCNN_DIR)
