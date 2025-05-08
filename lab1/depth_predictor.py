from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
from transformers import pipeline

from tqdm import tqdm


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    # Example for reading the image files in a folder
    path_rgb = 'lab1/data/livingroom1-color/'
    path_depth = 'lab1/data/livingroom1-depth-clean/'
    path_out = 'lab1/data/livingroom1-depth-predicted/'


    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "icp" # Only ICP
    separation = 1
    visual = False
    examples = -1
    ##########################################################################################################

    # Load model
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device=0)


    # READ GT TRajectories
    traj = read_trajectory('/home/xavi/master_code/aslam/lab1/data/livingroom1-traj.txt')
    examples = len(traj) if examples==-1 else examples
    for i in tqdm(range(2868, examples)): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        
        source, target, rgbd_source, rgbd_target = read_pointclouds(i+separation, i, path_depth)
        pred_target, pred_rgbd_target = get_depth_predictions(rgbd_target, pipe)

        # Save predicted depth images for source and target
        target_idx = i
        # Save target predicted depth
        depth_scale = 1000
        depth_target = np.asarray(pred_rgbd_target.depth) * depth_scale
        depth_target = o3d.geometry.Image(depth_target.astype(np.uint16))
        o3d.io.write_image(os.path.join(path_out, f"{target_idx:05d}.png"), depth_target)
        # o3d.io.write_image(os.path.join(path_out, f"{target_idx:05d}.png"), pred_rgbd_target.depth)