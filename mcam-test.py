from owl.instruments import MCAM
import numpy as np
import owl.mcam_data as mcam_data
import time 
import sys
import math
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
import csv
from wgpu.gui.auto import run
import pylinalg as la 
from controller import SurgicalController
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CROP_VALUES = (0.0, 1.0, 0.0, 1.0)
N_ITERATIONS = 6000
DEPTH_RANGE = (-20, 0)
EST_DEPTH = 8
N_CAMERAS_X = 8
N_CAMERAS_Y = 6
BINNING = 4
DOWNSAMPLING = 2
DOWNSAMPLING_AND_BINNING = BINNING * DOWNSAMPLING
CAMERA_ARRANGEMENT = [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
WARMUP_ITERATIONS = 150
ITERATIONS = 1
CALLIBRATION_FILE = 'data/other_calib_files/calibration_information'
CROPPING_XE = 128
CROPPING_YE = 128
CROPPING_XS = 128
CROPPING_YS = 128
LOWLIGHT_TEST_MS = 1e-3
LOWLIGHT_TEST = False
TEST = False
EXAG = 10
#FILTER_SIZE = (1, 5)
dlock = threading.Lock()
TIME_REFRESH_RATE = False
TIME_RECONSTRUCTION_RATE = False
 
'''def reset_view(event):
    if event.key == 'r':'''

def gaussian_kernel(size, sigma = 1, device = 'cuda:0'):
    x = torch.arange(size, device = device) - size // 2
    gaus_kern = torch.exp((x ** 2) / (2 * (sigma**2)))
    return (gaus_kern / gaus_kern.sum()).view(1, 1, -1)
 
def gaussian_blur(img, kernel_size, sigma):
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)

    k_1d = gaussian_kernel(kernel_size, sigma).to(img.device)
    padding = kernel_size // 2

    img = F.conv2d(img, k_1d.unsqueeze(2), padding=(0, padding))
    img = F.conv2d(img, k_1d.unsqueeze(3), padding=(padding, 0))

    return img

def go_home(camera):
    camera.local.position = (0, -768, 1000)
    camera.look_at((0, 0, 0))

def quit():
    global computed_data, mcam, canvas, canvas_img
    computed_data['run_manager'].end()
    canvas_img.close()
    canvas.close()
    print('Light Fury: killing the FiLMScope loop thread successfully...')
    return True 

def handle_event(event):
    global camera, mesh, points
    if event.type == "key_down":
        if event.key == "r":
            go_home(camera)
        if event.key == 'x':
            quit()
        if event.key == 'm':
            mesh.visible = True
            points.visible = False
        if event.key == 'p':
            mesh.visible = False
            points.visible = True
       

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
    sx, ex, sy, ey = saved_data['depth_crops']
    psx, pex, psy, pey = saved_data['post_crops']

    depths = depths[:, sx:ex, sy:ey]

    depths = F.interpolate(
        depths.unsqueeze(0),
        scale_factor = DOWNSAMPLING,
        mode = 'bilinear',
        align_corners = True
    )
    #depths = gaussian_blur(depths, 0, 0.25)
    depths = depths.squeeze(0).squeeze(0).cpu().numpy()[psx:pex, psy:pey] 
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
    sx, ex, sy, ey = saved_data['pre_crop_info']
    psx, pex, psy, pey = saved_data['post_crops']
    ref = color_ref[sx:ex, sy:ey][psx:pex, psy:pey]
   
    '''dem_e = time.perf_counter()
    print(f'Demosaicing Time: {dem_e - dem_s}')'''
    return ref

def animate():
    global computed_data, geometry, texture, dlock

    if TIME_REFRESH_RATE:
        global last_frame_time
        curr_time = time.perf_counter()
        duration = curr_time - last_frame_time
        print(1 / duration)
        last_frame_time = curr_time

    with dlock:
        if computed_data['new_data']:
            geometry.positions.data[:, 2] = np.flipud(computed_data['depth'])[y, x] * EXAG
            geometry.positions.update_range(0, geometry.positions.data.shape[0])
            texture.set_data(computed_data['reference'])
            computed_data['new_data'] = False
    renderer.render(scene, camera)
    controller.tick()
    canvas.request_draw()

def animate_viz():
    global computed_data, geometry, dlock
    with dlock:
        texture_img.set_data(computed_data['reference'])
    renderer_img.render(scene_img, camera_img)
    canvas_img.request_draw()

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

            startx, endx, starty, endy = run_manager.dataset.pre_crops[run_manager.dataset.reference_camera]
            sx, sy = startx * DOWNSAMPLING, starty * DOWNSAMPLING
            ex, ey = (endx + 1) * DOWNSAMPLING, (endy + 1) * DOWNSAMPLING

            saved_data['pre_crop_info'] = (sx, ex, sy, ey)

            startxp, endxp, startyp, endyp = run_manager.dataset.depth_crops[run_manager.dataset.reference_camera]

            if endyp < 0:
                endyp = endy - (endyp if endyp == -1 else endyp - 1)
            if endxp < 0:
                endxp = endx - (endxp if endxp == -1 else endxp - 1)

            saved_data['depth_crops'] = (startxp, endxp, startyp, endyp)
            saved_data['post_crops'] = (CROPPING_XS, -CROPPING_XE, CROPPING_YS, -CROPPING_YS)
            computed_data['run_manager'] = run_manager

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

        if TEST: 
            fig, ax = plt.subplots()
            img = ax.imshow(depths, cmap = 'turbo')
            ax.axis('off')
            plt.colorbar(img, ax=ax)
            plt.show()
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
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                if os.stat(filename).st_size == 0:
                    writer.writerow(headers)

                if computed_data['frame'] > 0:
                    new_row = [BINNING, DOWNSAMPLING, acq_dur, vol_dur, recon_dur, throughput, latency]
                    writer.writerow(new_row)

    if computed_data['frame'] >= N_ITERATIONS:
        quit()
        mcam.close()

# Script
mcam = MCAM()
mcam.exposure = 1e-3 if not LOWLIGHT_TEST else LOWLIGHT_TEST_MS
mcam.bin_mode = BINNING
mcam.digital_gain_color = [1.2, 1.05, 1.5]   # RGB
mcam.analog_gain = 1
# specify cameras from array to include
camera_array = np.zeros((N_CAMERAS_X, N_CAMERAS_Y), dtype = bool)
camera_array[2:-2, 1:-1] = True
#median_unfold = nn.Unfold(kernel_size = FILTER_SIZE)
headers = ['binning', 'subsampling', 'acq', 'vol', 'recon', 'throughput', 'latency']
filename = 'timing.csv'
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
        'height_est' : EST_DEPTH,           
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
    'new_data' : False,
    'current_data' : None
}

filmscope_loop()
H, W, _ = computed_data['reference'].shape

'''plt.imshow(computed_data['depth'], cmap = 'turbo')
plt.colorbar()
plt.show()
'''
if TIME_REFRESH_RATE:
    last_frame_time = time.perf_counter()

scene = gfx.Scene()
scene.add(gfx.AmbientLight(intensity = 25))
#scene.add(gfx.DirectionalLight())

geometry = gfx.plane_geometry(width = W, height = H, width_segments = W - 1, height_segments = H - 1)   
texture = gfx.Texture(computed_data['reference'], dim = 2)
pmaterial = gfx.PointsMaterial(map = texture)
mmaterial = gfx.MeshStandardMaterial(map = texture)
positions = geometry.positions.data
x = positions[:, 0]
y = positions[:, 1]
x = np.clip(np.round(x + np.abs(np.min(np.round(x)))).astype(int), 0, W - 1)
y = np.clip(np.round(y + np.abs(np.min(np.round(y)))).astype(int), 0, H - 1)
points = gfx.Points(geometry, pmaterial)
mesh = gfx.Mesh(geometry, mmaterial)
mesh.local.position = (0, 0, 0)
mesh.visible = False
points.local.position = (0, 0, 0)
points.visible = True
scene.add(mesh)
scene.add(points)
camera = gfx.PerspectiveCamera()
go_home(camera)

controller = SurgicalController(camera, H, W)
#controller.speed = 500.0
#print(camera.show_pos())

'''image.local.scale_y = -1
image.local.scale_x = -1
image.local.position = mesh.local.position.copy()
image.local.rotation = mesh.local.rotation.copy()
camera_img.world.up = camera.world.up'''

plane_size = max(W, H)
plane_geometry = gfx.plane_geometry(min(W, H), plane_size)
plane_xy = gfx.Mesh(plane_geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.15), side="both"))
#plane_yz = gfx.Mesh(plane_geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.15), side="both"))
#plane_xz = gfx.Mesh(plane_geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.15), side="both"))

#plane_xz.local.rotation = la.quat_from_euler((-np.pi / 2, 0, 0))
#plane_yz.local.rotation = la.quat_from_euler((0, -np.pi / 2, 0))
#plane_xz.local.position = (0, H / 2, 0)
#plane_yz.local.position = (-W / 2, 0, 0)
#scene.add(plane_xy) #, plane_xz, plane_yz) 
"""axis_positions = np.array(
    [[-W / 2, -H / 2, -H / 2], [W / 2, -H / 2, -H / 2],   # X
     [-W / 2, -H / 2, -H / 2], [-W / 2, H / 2, -H / 2],   # Y
     [-W / 2, -H / 2, -H / 2], [-W/ 2, -H / 2, H / 2],   # Z
    ], dtype=np.float32)

axis_colors = np.array([
    [1,0,0,1],[1,0,0,1],
    [0,1,0,1],[0,1,0,1],
    [0,0,1,1],[0,0,1,1],
], dtype=np.float32)"""

#geom = gfx.Geometry(positions=axis_positions, colors=axis_colors)
#scene.add(gfx.Line(geom, gfx.LineSegmentMaterial(thickness=3)))
#scene.local.rotation = la.quat_from_axis_angle((0, 0, 1), np.radians(90))

compute_thread = threading.Thread(target = filmscope_loop, daemon=True)
compute_thread.start()

import glfw

canvas = RenderCanvas(size = (1280, 1080), title = 'FiLMScope 2.5D Reconstruction')
glfw.set_window_pos(canvas._window, 640, 0)

renderer = gfx.WgpuRenderer(canvas)
renderer.add_event_handler(handle_event, "key_down")
controller.register_events(renderer)
canvas.request_draw(animate)

# reference image 
canvas_img = RenderCanvas(size = (575, 1080), title = 'Live Images')
glfw.set_window_pos(canvas_img._window, 0, 0)
camera_img = gfx.OrthographicCamera(zoom = 1.25)
scene_img = gfx.Scene()
texture_img = gfx.Texture(computed_data['reference'], dim = 2)
image = gfx.Image(
    gfx.Geometry(grid=texture_img), 
    gfx.ImageBasicMaterial(clim=(0, 255))
)
image.local.scale = (1., -1., 1.)
angle_rad = math.radians(90)
image.local.rotation = la.quat_from_euler((0, 0, angle_rad), order="XYZ")
scene_img.add(image)
scene_img.add(gfx.AmbientLight(intensity = 1))
camera_img.show_object(scene_img)
renderer_img = gfx.renderers.WgpuRenderer(canvas_img)
canvas_img.request_draw(animate_viz)

run() 
