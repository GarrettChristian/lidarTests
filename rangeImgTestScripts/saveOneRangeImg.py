


from laserVisRangeImage import LaserScan
from laserVisRangeImage import LaserScanVis
import numpy as np
import glob, os


# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/testScriptsKitti/unrealistic1.bin"
# binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"
binFileName = "removeHalfByYval.bin"

# Create vis object and scan with an arbitrary pcd
scan = LaserScan(project=True)
vis = LaserScanVis(scan, binFileName)


fileName = os.path.basename(binFileName)
fileName = fileName.replace(".bin", "")
saveAt = fileName + ".png"

            
# vis.set_new_pcd(binFileName)
vis.save(saveAt)
print("saved", saveAt)
