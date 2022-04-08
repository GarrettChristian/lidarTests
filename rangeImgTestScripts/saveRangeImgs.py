


from laserVisRangeImage import LaserScan
from laserVisRangeImage import LaserScanVis
import numpy as np
import glob, os


path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/"
binFileName = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/000000.bin"

# Set up directories if they don't exist
isExist = os.path.exists("rangeimgs")
if not isExist:
  os.makedirs("rangeimgs")
  
for x in range(0, 1):
    folderNum = str(x).rjust(2, '0')
    isExist = os.path.exists("rangeimgs/" + folderNum)
    if not isExist:
        os.makedirs("rangeimgs/" + folderNum)

# Create vis object and scan with an arbitrary pcd
scan = LaserScan(project=True)
vis = LaserScanVis(scan, binFileName)

for x in range(0, 1):
    
    folderNum = str(x).rjust(2, '0')
    currPath = path + folderNum

    files = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
    print("parsing ", currPath)

    i = 0
    
    for file in files:

        fileName = os.path.basename(file)
        fileName = fileName.replace(".bin", "")
        
        saveAt = "rangeimgs/" + folderNum + "/" + fileName + ".png"
        isExist = os.path.exists(saveAt)
        if not isExist:
            
            vis.set_new_pcd(file)
            vis.save(saveAt)
            print(i, saveAt)
            # except:
            #     failedFiles.append(file)
            #     print(file)
            #     print(saveAt)
            i += 1
