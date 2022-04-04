"""
https://github.com/PRBonn/lidar-bonnetal/blob/5a5f4b180117b08879ec97a3a05a3838bce6bb0f/train/common/laserscan.py
Lovingly borrowing & modfiying from: RangeNet++: Fast and Accurate LiDAR Semantic Segmentation


Not worth it rn

"""

import numpy as np

import glob, os
import struct
import open3d as o3d
from vispy.scene import visuals, SceneCanvas
from vispy import scene
from vispy import app
from vispy.io import load_data_file, read_png


binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
np_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(np_arr))

np_arr = np_arr.reshape((int(np.shape(np_arr)[0]) // 4, 4))
print(np.shape(np_arr))
pcd = np.delete(np_arr, 3, 1)
print(np.shape(pcd))


pcd
proj_H=64
proj_W=1024

""" Project a pointcloud into a spherical projection image.projection.
    Function takes no arguments because it can be also called externally
    if the value of the constructor was not set (in case you change your
    mind about wanting the projection)
"""
proj_range = np.full((proj_H, proj_W), -1,dtype=np.float32)

# laser parameters
fov_up = 3.0 / 180.0 * np.pi      # field of view up in rad
fov_down = -25.0 / 180.0 * np.pi  # field of view down in rad
fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

# get depth of all points
depth = np.linalg.norm(pcd, 2, axis=1)

# get scan components
scan_x = pcd[:, 0]
scan_y = pcd[:, 1]
scan_z = pcd[:, 2]

# get angles of all points
yaw = -np.arctan2(scan_y, scan_x)
pitch = np.arcsin(scan_z / depth)

# get projections in image coords
proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

# scale to image size using angular resolution
proj_x *= proj_W                              # in [0.0, W]
proj_y *= proj_H                              # in [0.0, H]

# round and clamp for use as index
proj_x = np.floor(proj_x)
proj_x = np.minimum(proj_W - 1, proj_x)
proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
# self.proj_x = np.copy(proj_x)  # store a copy in orig order

proj_y = np.floor(proj_y)
proj_y = np.minimum(proj_H - 1, proj_y)
proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
# self.proj_y = np.copy(proj_y)  # stope a copy in original order

# copy of depth in original order
# self.unproj_range = np.copy(depth)

# order in decreasing depth
indices = np.arange(depth.shape[0])
order = np.argsort(depth)[::-1]
depth = depth[order]
indices = indices[order]
points = pcd[order]
# remission = self.remissions[order]
proj_y = proj_y[order]
proj_x = proj_x[order]

# assing to images
proj_range[proj_y, proj_x] = depth
# proj_xyz[proj_y, proj_x] = points
# proj_remission[proj_y, proj_x] = remission
# proj_idx[proj_y, proj_x] = indices
# proj_mask = (self.proj_idx > 0).astype(np.int32)


# https://vispy.org/gallery/scene/image.html

canvas = scene.SceneCanvas(keys='interactive')
canvas.size = proj_W, proj_H
canvas.show()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

img_vis = visuals.Image(cmap='viridis')
data = np.copy(proj_range)
data[data > 0] = data[data > 0]**(1 / 16)
data[data < 0] = data[data > 0].min()
data = (data - data[data > 0].min()) / \
    (data.max() - data[data > 0].min())
img_vis.set_data(data)
img_vis.update()

image = img_vis

canvas.title = 'Spatial Filtering using '

# Set 2D camera (the camera will scale to the contents in the scene)
view.camera = scene.PanZoomCamera(aspect=1)
# flip y-axis to have correct aligment
view.camera.flip = (0, 1, 0)
view.camera.set_range()
view.camera.zoom(0.1, (250, 200))

# get interpolation functions from Image
names = image.interpolation_functions
names = sorted(names)
act = 17

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global act
    if event.key in ['Left', 'Right']:
        if event.key == 'Right':
            step = 1
        else:
            step = -1
        act = (act + step) % len(names)
        interpolation = names[act]
        image.interpolation = interpolation
        canvas.title = 'Spatial Filtering using %s Filter' % interpolation
        canvas.update()

app.run()

