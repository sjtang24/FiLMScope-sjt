from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

directory = Path('.')

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem) 

def clean(directory, pathname):
    print(f"Deleting {pathname}...")
    if pathname != "goldstandards":
        newfilename = "-".join("_".join(pathname.split('/')).split(' '))
    else:
        newfilename = pathname
    all_files = sorted(directory.glob(f'*.npy'), key = numeric_sort_key)
    
    if len(all_files) == 0:
        return

    recon_frames = None
    for ifile, file in enumerate(all_files):
        data = np.load(file, mmap_mode='r')[-1, :, :]
        if ifile == 0:
            w, h = data.shape
            recon_frames = np.zeros((len(all_files), w, h))
        recon_frames[ifile, :, :] = data
        os.remove(file)
    np.save(f'{newfilename}.npy', recon_frames)
    
for iterations in [10, 20, 30, 'goldstandards']:
    if iterations == 'goldstandards':
        directory = Path(f"{iterations}")
        clean(directory, "goldstandards")
    else: 
        for arrangements in ['2x2 grid', '4x4 grid', 'all_cameras', 'narrow_sparse', 'wide_sparse']:
            for downsampling in [1, 2, 4, 8]: 
                pathname = f'{iterations}/{arrangements}/{downsampling}'
                directory = Path(pathname)
                if directory.exists():
                    clean(directory, pathname)
                    print(f"Removing Directory {iterations}/{arrangements}/{downsampling}...")
                    os.rmdir(f"{iterations}/{arrangements}/{downsampling}")
            print(f"Removing Directory {iterations}/{arrangements}...")
            os.rmdir(f"{iterations}/{arrangements}")
    print(f"Removing Directory {iterations}...")
    os.rmdir(f"{iterations}")
    

