import os
import sys

# Add the path of Mask_RCNN to use it as a submodule.
DPN_DIR = os.path.abspath(os.path.join(__path__[0], ".."))
if DPN_DIR not in sys.path:  # Check if the path is already on the search path.
    sys.path.append(DPN_DIR)

# Add the path of Mask_RCNN to use it as a submodule.
MRCNN_DIR = os.path.abspath(os.path.join(__path__[0], "..", "external", "Mask_RCNN"))
if MRCNN_DIR not in sys.path:  # Check if the path is already on the search path.
    sys.path.append(MRCNN_DIR)
