import pygfx as gfx
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

# From https://github.com/pygfx/pygfx/issues/836#issuecomment-2272664027

# Copy from https://github.com/vispy/vispy/blob/main/vispy/util/filter.py
def gaussian_filter(data, sigma):
    """
    Drop-in replacement for scipy.ndimage.gaussian_filter.

    (note: results are only approximately equal to the output of
     gaussian_filter)
    """
    if np.isscalar(sigma):
        sigma = (sigma,) * data.ndim

    baseline = data.mean()
    filtered = data - baseline
    for ax in range(data.ndim):
        s = float(sigma[ax])
        if s == 0:
            continue

        # generate 1D gaussian kernel
        ksize = int(s * 6)
        x = np.arange(-ksize, ksize)
        kernel = np.exp(-x**2 / (2*s**2))
        kshape = [1, ] * data.ndim
        kshape[ax] = len(kernel)
        kernel = kernel.reshape(kshape)

        # convolve as product of FFTs
        shape = data.shape[ax] + ksize
        scale = 1.0 / (abs(s) * (2*np.pi)**0.5)
        filtered = scale * np.fft.irfft(np.fft.rfft(filtered, shape, axis=ax) *
                                        np.fft.rfft(kernel, shape, axis=ax),
                                        axis=ax)

        # clip off extra data
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(filtered.shape[ax]-data.shape[ax], None, None)
        filtered = filtered[tuple(sl)]
    return filtered + baseline


# Create the surface geometry
# Reference: https://github.com/pygfx/pygfx/blob/8661e8f69037c0480da72c983038e4553b4362ca/pygfx/geometries/_plane.py#L9-L39

# x = np.arange(-20, 20, 0.25)
# y = np.arange(-20, 20, 0.25)
# xx, yy = np.meshgrid(x, y)
# zz = np.sin(np.sqrt(xx**2 + yy**2))
# xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()

x = np.linspace(0, 250, 250, dtype=np.float32)
y = np.linspace(0, 250, 250, dtype=np.float32)
x, y = np.meshgrid(x, y)
z = np.random.normal(size=(250, 250), scale=200)
z[100, 100] += 50000
z = gaussian_filter(z, (10, 10))
xx, yy, zz = x.flatten(), y.flatten(), z.flatten()

positions = np.column_stack([xx, yy, zz]).astype(np.float32)

w, h = x.max() - x.min(), y.max() - y.min()
dim = np.array([w, h], dtype=np.float32)
texcoords = (positions[..., :2] + dim / 2) / dim
texcoords[..., 1] = 1 - texcoords[..., 1]

ny = len(y)
nx = len(x)
# the amount of vertices
indices = np.arange(ny * nx, dtype=np.uint32).reshape((ny, nx))
# for every panel (height_segments, width_segments) there is a quad (2, 3)
index = np.empty((ny-1, nx-1, 2, 3), dtype=np.uint32)
# create a grid of initial indices for the panels
index[:, :, 0, 0] = indices[
    np.arange(ny-1)[:, None], np.arange(nx-1)[None, :]
]
# the remainder of the indices for every panel are relative
index[:, :, 0, 1] = index[:, :, 0, 0] + 1
index[:, :, 0, 2] = index[:, :, 0, 0] + nx
index[:, :, 1, 0] = index[:, :, 0, 0] + nx + 1
index[:, :, 1, 1] = index[:, :, 1, 0] - 1
index[:, :, 1, 2] = index[:, :, 1, 0] - nx

index = index.reshape((-1, 3))  

# Compute face normals
a = positions[index[:, 0]]
b = positions[index[:, 1]]
c = positions[index[:, 2]]

face_normals = np.cross(b - a, c - a)
face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

# Compute vertex normals
normals = np.zeros_like(positions)
np.add.at(normals, index[:, 0], face_normals)
np.add.at(normals, index[:, 1], face_normals)
np.add.at(normals, index[:, 2], face_normals)

normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

g = gfx.Geometry(positions=positions, indices=index, texcoords=texcoords, normals=normals)

surface = gfx.Mesh(g, gfx.MeshPhongMaterial(color=(0.3, 0.3, 1), shininess=0))

app = QApplication(sys.argv)
gfx.show(surface)
app.exec_()
