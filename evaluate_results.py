import matplotlib.pyplot as plt
import pickle
import numpy as np 

TOTAL_ITERS = 25
ITERS = 3
with open(f'timing-results_{ITERS}.pkl', 'rb') as timing_file:
    timing_dict = pickle.load(timing_file)

print(timing_dict)
capture_time = np.array(timing_dict['capture'])
network_time = np.array(timing_dict['frame_time'])
setup_time = np.array(timing_dict['setup'])
postproc_time = np.array(timing_dict['post_processing'])
viz_time = np.array(timing_dict['visualization'])

frame = np.arange(TOTAL_ITERS)
"""
create_ds_time = np.array(timing_dict['setup_inner']['create-dataset'] + [0] * (TOTAL_ITERS - 1))
batch_time = np.array(timing_dict['setup_inner']['batching'] + [0] * (TOTAL_ITERS - 1))
t2cpu_time = np.array([0] + timing_dict['swap_info']['transfer-cpu'])
swap_time = np.array([0] + timing_dict['swap_info']['swap-frames'])
prepvol_time = np.array([timing_dict['setup_inner']['prep-volume']] + timing_dict['swap_info']['prep-volume'])
t2gpu_time = np.array([timing_dict['setup_inner']['transfer-gpu']] + timing_dict['swap_info']['transfer-gpu'])
"""
"""
img_prep_times = np.array(timing_dict['swap_frames_info']['image-prep'])
img_crop_times = np.array(timing_dict['swap_frames_info']['image-crop'])
"""
"""
img_prep_load_times = np.array(timing_dict['prep_info']['load'])
img_prep_copy_times = np.array(timing_dict['prep_info']['copy'])
"""
plt.bar(frame, capture_time, label = 'Image Capturing')
plt.bar(frame, setup_time, bottom = capture_time, label = 'Loading and Transferring')
plt.bar(frame, network_time, bottom = capture_time + setup_time, label = 'In Network')
plt.bar(frame, postproc_time, bottom = capture_time + setup_time + network_time, label = 'Post-Processing')
plt.bar(frame, viz_time, bottom = capture_time + setup_time + network_time + postproc_time, label = 'Visualization (Matplotlib)')

plt.legend(title = 'MCAM-Reconstruction Loop Subroutines', ncol = 1, loc = 'upper center')
plt.xlabel('Frame Number')
plt.ylabel('Time per Iteration (in Seconds)')
plt.title('An Overview of Duration Spent per Iteration')
plt.show()


"""
plt.bar(frame, create_ds_time, label = 'Dataset Creating')
plt.bar(frame, batch_time, bottom = create_ds_time, label = 'Batching')
plt.bar(frame, t2cpu_time, bottom = create_ds_time + batch_time, label = 'Transferring (CPU)')
plt.bar(frame, swap_time, bottom = create_ds_time + batch_time + t2cpu_time, label = 'Swapping Images')
plt.bar(frame, prepvol_time, bottom = create_ds_time + batch_time + t2cpu_time + swap_time, label = 'Preparing Volume')
plt.bar(frame, t2gpu_time, bottom = create_ds_time + batch_time + t2cpu_time + swap_time + prepvol_time, label = 'Transferring (GPU)')
plt.legend(title = 'RunManager Subroutines', ncol = 1, loc = 'upper center')
plt.xlabel('Frame Number')
plt.ylabel('Time per Iteration (in Seconds)')
plt.title('Time in RunManager Subroutines')
plt.show()
"""
"""
plt.bar(frame, img_prep_load_times, label = 'Image Loading')
plt.bar(frame, img_prep_copy_times, bottom = img_prep_load_times, label = 'Image Copying')
plt.legend(title = 'Image Preparation Subroutines', ncol = 1, loc = 'lower center')
plt.xlabel('Frame Number')
plt.ylabel('Time per Iteration (in Seconds)')
plt.title('Time in Subroutines for Preparing Images')
plt.ylim(7.5, 8.2)
plt.show()

plt.bar(frame, img_prep_times, label = 'Image Preparation')
plt.bar(frame, img_crop_times, bottom = img_prep_times, label = 'Image Cropping')
plt.legend(title = 'Initializing and Swapping Images Subroutines', ncol = 1, loc = 'lower center')
plt.xlabel('Frame Number')
plt.ylabel('Time per Iteration (in Seconds)')
plt.title('Time in Subroutines for FSDataset Image Swapping and Initializing')
plt.ylim(7, 9)
plt.show()
"""