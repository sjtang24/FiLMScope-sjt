import pickle
import os
import plotly.graph_objects as go
import numpy as np

maps_3d_dir = f"plots/3dheightmap"
try:
    os.makedirs(maps_3d_dir)
except FileExistsError:
    pass

with open(f"data/gold-standards.pkl", "rb") as gold_standards:
    gstds = pickle.load(gold_standards)

for frame in gstds.keys():
    Z = gstds[frame]
    img_h, img_w = Z.shape
    X, Y = np.meshgrid(np.arange(img_w), np.arange(img_h))

    fig = go.Figure(data=[go.Surface(z=Z, x=-Y, y=-X, colorscale='Blues')])
    fig.update_layout(
        width = 900, height = 900,
        scene_camera=dict(eye=dict(x=0, y=0, z=2)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )    
    )

    fig.write_image(f"data/3d-{frame}.png")
