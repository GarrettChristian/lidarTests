


from laserVisRangeImage import LaserScan
from laserVisRangeImage import LaserScanVis

# create a visualizer
scan = LaserScan(project=True)
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/velodyne/000000.bin"
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/velodyne/000326.bin"
vis = LaserScanVis(scan=scan, scan_name=binFileName)

vis.run()