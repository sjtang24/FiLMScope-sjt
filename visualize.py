from PIL import Image
import os 
import pickle
import numpy as np 

TOTAL_ITERS = 25
MS_PER_S = 1000
SPEEDUP = 4

with open('timing-results.pkl', 'rb') as timing_file:
    timing_dict = pickle.load(timing_file)

capture_time = np.array(timing_dict['capture'])
network_time = np.array(timing_dict['frame_time'])
setup_time = np.array(timing_dict['setup'])
postproc_time = np.array(timing_dict['post_processing'])

frame = np.arange(len(postproc_time))
total_time = capture_time + network_time + setup_time + postproc_time
total_time = total_time[1:]
total_time_in_sec = (np.append(total_time, 1) * MS_PER_S / SPEEDUP).astype(int).tolist()
print(total_time_in_sec)
def extract_number(filename):
    stem, _ = os.path.splitext(filename)
    return int(stem.split('frame')[1])

RUN = 0
folder_path = f'recons/Run{RUN}'
filenames = os.listdir(folder_path)

images_files = sorted(filenames, key=extract_number)
frames = [Image.open(folder_path + '/' + img) for img in images_files]
frames[0].save(
    f'reconstruction_run{RUN}.gif',
    format='GIF',
    save_all=True,
    append_images=frames[1:],
    duration=total_time_in_sec,
    loop=1
)
