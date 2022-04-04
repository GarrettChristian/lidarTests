from __future__ import annotations
import json

from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='./../data/sets/nuscenes', verbose=True)

# Catergories
print(nusc.list_lidarseg_categories(sort_by='count'))

# Specific sample from the 10
my_sample = nusc.sample[1]
print(json.dumps(my_sample, sort_keys=True, indent=4))


sensor = 'LIDAR_TOP'
lidarData = nusc.get('sample_data', my_sample['data'][sensor])
# "filename": "samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.pcd.bin",

# Render data top view
# nusc.render_sample_data(lidarData['token'])
# Render top view with multiple sweeps included
# nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)

print("-----------------------------------")
print(json.dumps(lidarData, sort_keys=True, indent=4))

annotations = nusc.sample_annotation[0]

print("-----------------------------------")
print(json.dumps(annotations, sort_keys=True, indent=4))

# Render with image
# nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=True)

# All images
# my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
# nusc.render_scene(my_scene_token)




