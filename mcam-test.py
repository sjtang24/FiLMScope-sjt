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
N_ITERATIONS = 150
DEPTH_RANGE = (-15, -5)
N_CAMERAS_X = 8
N_CAMERAS_Y = 6
DOWNSAMPLING = 4
CAMERA_ARRANGEMENT = [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
WARMUP_ITERATIONS = 150
ITERATIONS = 1
CALLIBRATION_FILE = 'data/calibration_information'
CROPPING_XE = 1024
CROPPING_YE = 64
CROPPING_XS = 256
CROPPING_YS = 64
LOWLIGHT_TEST_MS = 150e-3
LOWLIGHT_TEST = True
TEST = False
#FILTER_SIZE = (1, 5)
dlock = threading.Lock()
TIME_REFRESH_RATE = False
TIME_RECONSTRUCTION_RATE = False

def gaussian_kernel(size, sigma = 1, device = 'cuda:0'):
    x = torch.arange(size, device = device) - size // 2
    gaus_kern = torch.exp((x ** 2) / (2 * (sigma**2)))
    return gaus_kern / gaus_kern.sum()


def recontruct_depth():
    global saved_data, computed_data
    '''torch.cuda.synchronize()
    d_s = time.perf_counter()'''
    frame_no = computed_data['frame']
    run_manager = computed_data['run_manager'] # safe because only modified sequentially between reconstruct and init/swap
    config_dict = saved_data['config_dict']
    iters = config_dict["run_args"]["iters"] if frame_no == 0 else ITERATIONS

    for j in tqdm(range(iters), disable=(not TEST)):
        log = (j % config_dict["run_args"]["display_freq"] == 0) or (j == iters - 1)
 
        _, _, _, outputs, loss = run_manager.run_epoch(
            j, log=(log and config_dict["use_neptune"])
        )
        
    depths = outputs["depth"].detach()

    depths = F.interpolate(
        depths.unsqueeze(0),
        scale_factor = DOWNSAMPLING,
        mode = 'bilinear',
        align_corners = True
    )

    depths = depths.squeeze(0).squeeze(0).cpu().numpy()  
    sx, ex, sy, ey = saved_data['cropping_info']
    depths =  depths[sy:ey, sx:ex]
    '''torch.cuda.synchronize()
    d_e = time.perf_counter()
    print(f'Depth Reconstruction Time: {d_e - d_s}')'''
    return depths

def demosaic(colored_ref): 
    global saved_data
    '''torch.cuda.synchronize()
    dem_s = time.perf_counter()'''
    cref = np.rot90(colored_ref, k = 1)
    color_ref = cv2.cvtColor(cref, cv2.COLOR_BAYER_RG2RGB)
    sx, ex, sy, ey = saved_data['cropping_info']
    ref = color_ref[sy:ey, sx:ex]
    '''dem_e = time.perf_counter()
    print(f'Demosaicing Time: {dem_e - dem_s}')'''
    return ref

def animate():
    global computed_data, canvas, geometry, texture

    if TIME_REFRESH_RATE:
        global last_frame_time
        curr_time = time.perf_counter()
        duration = curr_time - last_frame_time
        print(duration)
        last_frame_time = curr_time

    if dlock.acquire(blocking = False):
        try:
            if computed_data['new_data']:
                geometry.positions.data[:, 2] = np.flipud(computed_data['depth'])[y, x] * 150
                geometry.positions.update_range(0, geometry.positions.data.shape[0])
                texture.set_data(computed_data['reference'])
                computed_data['new_data'] = False
        finally:
            dlock.release()
    

def filmscope_loop():
    global saved_data, computed_data, canvas
    mcam = saved_data['mcam']
    camera_array = saved_data['camera_array']
    config_dict = saved_data['config_dict']

    iters = N_ITERATIONS - 1 if computed_data['frame'] > 0 else 1
    for _ in range(iters):
        if TIME_RECONSTRUCTION_RATE:
            s_acq = time.perf_counter()

        dset = mcam.acquire_selection(camera_array)

        if TIME_RECONSTRUCTION_RATE:
            e_acq = time.perf_counter()
            acq_dur = e_acq - s_acq
            s_vol = time.perf_counter()

        if computed_data['frame'] == 0: #initialize run manager
            # doesn't matter if run_manager will cause race conditions because 
            # we exclusively run swap/init or perform the reconstruction
            run_manager = RunManager(
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
            saved_data['cropping_info'] = (sx, ex, sy, ey)

        else:
            computed_data['run_manager'].swap_frames(sample_image=dset)

        if TIME_RECONSTRUCTION_RATE:
            torch.cuda.synchronize()
            e_vol = time.perf_counter()
            vol_dur = e_vol - s_vol
            s_recon = time.perf_counter()
        
        colored_ref = computed_data['run_manager'].dataset.colored_ref

        # To obtain the colored image, we perform demosaicing (another experiment
        # where demosaic vs no-demosaic) in parallel with 3d reconstruction
        # This shaves about 0.04 seconds
        with ThreadPoolExecutor(max_workers = 2) as parallel_ex:
            heightmap_op = parallel_ex.submit(recontruct_depth)
            color_img_op = parallel_ex.submit(demosaic, colored_ref)

            depths = heightmap_op.result()
            refs = color_img_op.result()

        with dlock:
            computed_data['depth'] = depths
            computed_data['reference'] = refs
            computed_data['new_data'] = True
            computed_data['frame'] += 1 

        if TIME_RECONSTRUCTION_RATE:
            torch.cuda.synchronize()
            e_recon = time.perf_counter()
            recon_dur = e_recon - s_recon
            latency = acq_dur + vol_dur + recon_dur
            throughput = 1 / latency
            print(f'Frame #{computed_data['frame']} | ACQ: {acq_dur} + VOL: {vol_dur} + RECON: {recon_dur} | Throughput: {throughput} | Latency : {latency}')

    if computed_data['frame'] >= N_ITERATIONS:
        computed_data['run_manager'].end()
        mcam.close()
        QMetaObject.invokeMethod(canvas, 'close', Qt.QueuedConnection)
        print('Light Fury: killing the FiLMScope loop thread successfully...')

# Script
mcam = MCAM()
mcam.exposure = 1e-3 if not LOWLIGHT_TEST else LOWLIGHT_TEST_MS
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

saved_data = {
    'gauss_kernel' : gaussian_kernel(7),
    'config_dict' : config_dict, 
    'camera_array' : camera_array, 
    'mcam' : mcam, 
    'cropping_info' : None,
}

computed_data = {
    'frame' : 0, 
    'reference' : None, 
    'depth' : None, 
    'run_manager' : None,
    'new_data' : False
}

filmscope_loop()
if TIME_REFRESH_RATE:
    last_frame_time = time.perf_counter()
scene = gfx.Scene()
scene.add(gfx.AmbientLight(intensity = 5))
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


    