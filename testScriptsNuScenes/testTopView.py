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


print("-----------------------------------")
print(json.dumps(lidarData, sort_keys=True, indent=4))

nusc.render_sample_data(lidarData['token'])