from owl.instruments import MCAM
import numpy as np
import owl.mcam_data as mcam_data
import time 
import sys
from tqdm import tqdm 
import os
from filmscope.reconstruction import RunManager, generate_config_dict
import matplotlib.pyplot as plt 
import matplotlib.colors as pclr
import torch
import torch.nn as nn
import torch.nn.functional as F
from filmscope.util.filters_pt import median_filter
import pickle
from datetime import datetime
import cv2
import pygfx as gfx
from rendercanvas.auto import RenderCanvas
from PySide6.QtCore import QMetaObject, Qt
import threading 
from concurrent.futures import ThreadPoolExecutor


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CROP_VALUES = (0.0, 1.0, 0.0, 1.0)
N_ITERATIONS = 100
DEPTH_RANGE = (-5, 5)
N_CAMERAS_X = 8
N_CAMERAS_Y = 6
DOWNSAMPLING = 8
CAMERA_ARRANGEMENT = [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
WARMUP_ITERATIONS = 150
ITERATIONS = 1
CALLIBRATION_FILE = 'data/phantom/calibration_information'
CROPPING_XE = 1024
CROPPING_YE = 64
CROPPING_XS = 256
CROPPING_YS = 64
#FILTER_SIZE = (1, 5)


def gaussian_kernel(size, sigma = 1, device = 'cuda:0'):
    x = torch.arange(size, device = device) - size // 2
    gaus_kern = torch.exp((x ** 2) / (2 * (sigma**2)))
    return gaus_kern / gaus_kern.sum()


def recontruct_depth():
    global computed_data
    run_manager = computed_data['run_manager']
    config_dict = computed_data['config_dict']
    iters = config_dict["run_args"]["iters"] if computed_data['frame'] == 0 else ITERATIONS

    for j in range(iters):
        log = (j % config_dict["run_args"]["display_freq"] == 0) or (j == iters - 1)
 
        _, _, _, outputs, _ = run_manager.run_epoch(
            j, log=(log and config_dict["use_neptune"])
        )
        if j != iters - 1:
            continue

    depths = outputs["depth"].detach()

    depth_upsampled = F.interpolate(
        depths.unsqueeze(0),
        scale_factor = DOWNSAMPLING,
        mode = 'bilinear',
        align_corners = True
    )


    gauss = computed_data['gauss_kernel']
    kernel_size = gauss.shape[0]
    pad_s = (kernel_size - 1) // 2
    pad_e = (kernel_size - 1) - pad_s
  
    depth_upsampled = F.pad(depth_upsampled, (pad_s, pad_e, 0, 0))
    d_horz = F.conv2d(depth_upsampled, gauss.view(1, 1, 1, kernel_size))
    d_horz = F.pad(d_horz, (0, 0, pad_s, pad_e))
    d_vert = F.conv2d(d_horz, gauss.view(1, 1, kernel_size, 1))

    depth = d_vert.squeeze(0).squeeze(0).cpu().numpy()  
    sx, ex, sy, ey = computed_data['cropping_info']
    computed_data['depth'] = depth[sy:ey, sx:ex]

def demosaic(): 
    global computed_data
    run_manager = computed_data['run_manager']
    color_ref = cv2.cvtColor(run_manager.dataset.colored_ref, cv2.COLOR_BAYER_RG2RGB)
    sx, ex, sy, ey = computed_data['cropping_info']
    computed_data['reference'] = color_ref[sy:ey, sx:ex]

def animate():
    global computed_data, canvas, geometry, texture
    geometry.positions.data[:, 2] = np.flipud(computed_data['depth'])[y, x] * 150
    geometry.positions.update_range(0, geometry.positions.data.shape[0])
    texture.set_data(computed_data['reference'])

def filmscope_loop():
    global computed_data, canvas
    mcam = computed_data['mcam']
    camera_array = computed_data['camera_array']
    config_dict = computed_data['config_dict']
    iters = N_ITERATIONS - 1 if computed_data['frame'] > 0 else 1
    for _ in range(iters):
        # 0.073
        dset = mcam.acquire_selection(camera_array)
        
        # 0.052
        if computed_data['frame'] == 0: #initialize run manager
            run_manager = RunManager(
                {},
                config_dict, 
                sample = dset, 
                calibration_file=CALLIBRATION_FILE
            )

            indices = run_manager.dataset.full_crops[run_manager.dataset.reference_camera]

            startx, endx, starty, endy = tuple([
                -crop_coords if crop_coords < 0 else crop_coords
                for crop_coords in indices
            ])

            sx, sy = startx * DOWNSAMPLING + CROPPING_XS, starty * DOWNSAMPLING + CROPPING_YS
            ex, ey = endx * DOWNSAMPLING - CROPPING_XE, endy * DOWNSAMPLING - CROPPING_YE
            computed_data['run_manager'] = run_manager
            computed_data['cropping_info'] = (sx, ex, sy, ey)
        else:
            computed_data['run_manager'].swap_frames(sample_image=dset)

        # 0.27 seconds 
        # To obtain the colored image, we perform demosaicing (another experiment
        # where demosaic vs no-demosaic) in parallel with 3d reconstruction
        # This shaves about 0.04 seconds
        with ThreadPoolExecutor(max_workers = 2) as parallel_ex:
            heightmap_op = parallel_ex.submit(recontruct_depth)
            color_img_op = parallel_ex.submit(demosaic)

            heightmap_op.result()
            color_img_op.result()

        computed_data['frame'] += 1    

    if computed_data['frame'] >= N_ITERATIONS:
        computed_data['run_manager'].end()
        mcam.close()
        QMetaObject.invokeMethod(canvas, 'close', Qt.QueuedConnection)
        print('Falafel: killing the FiLMScope loop thread successfully...')

# Script
mcam = MCAM()
mcam.exposure = 1e-3
# specify cameras from array to include
camera_array = np.zeros((N_CAMERAS_X, N_CAMERAS_Y), dtype = bool)
camera_array[2:-2, 1:-1] = True
#median_unfold = nn.Unfold(kernel_size = FILTER_SIZE)


# loop would start here, and can be used in the algorithm 
config_dict = generate_config_dict(
    gpu_number = '0', downsample = DOWNSAMPLING, 
    camera_set = "custom", custom_image_numbers = CAMERA_ARRANGEMENT, 
    run_args = {
        'iters' : WARMUP_ITERATIONS,
        'batch_size' : 12,
        'num_depths' : 32, 
        'display_freq' : 1
    },
    custom_crop_info={
        'depth_range' : DEPTH_RANGE,  
        'height_est' : 5,           
        'crop_size' : (1, 1),       
        'ref_crop_center' : (0.5, 0.5)
    },
    crop_values = CROP_VALUES
)

computed_data = {'frame' : 0, 'reference' : None, 'depth' : None, 'run_manager' : None, 'gauss_kernel' : gaussian_kernel(11),
                 'config_dict' : config_dict, 'camera_array' : camera_array, 'mcam' : mcam, 'cropping_info' : None}

filmscope_loop()
scene = gfx.Scene()
scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight())

camera = gfx.PerspectiveCamera(70, 16/9)
H, W, _ = computed_data['reference'].shape
geometry = gfx.plane_geometry(width = W, height = H, width_segments = W - 1, height_segments = H - 1)   
texture = gfx.Texture(computed_data['reference'], dim = 2)
material = gfx.MeshPhongMaterial(map = texture)
positions = geometry.positions.data
x = positions[:, 0]
y = positions[:, 1]
x = np.clip(np.round(x + np.abs(np.min(np.round(x)))).astype(int), 0, W - 1)
y = np.clip(np.round(y + np.abs(np.min(np.round(y)))).astype(int), 0, H - 1)
mesh = gfx.Mesh(geometry, material)
mesh.local.position = (0, 0, 0)
scene.add(mesh)

compute_thread = threading.Thread(target = filmscope_loop, daemon=True)
compute_thread.start()


canvas = RenderCanvas(size = (1920, 1080), title = 'FiLMScope 2.5D Reconstruction')
renderer = gfx.renderers.WgpuRenderer(canvas)
gfx.show(scene, before_render = animate, renderer = renderer)


    