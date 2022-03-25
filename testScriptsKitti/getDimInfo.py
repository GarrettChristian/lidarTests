
import glob, os
import sys

minSize = sys.maxsize
maxSize = sys.maxsize * -1
avgSize = 0
count = 0

path = "/Volumes/Extreme SSD/semKitti/dataset/sequences/"
for file in glob.glob(path + "**/*.bin", recursive = True):
    # print(file)
    file_size = os.stat(file)
    maxSize = max(maxSize, file_size.st_size)
    minSize = min(minSize, file_size.st_size)
    count += 1
    avgSize += file_size.st_size
    # print("Size of file :", file_size.st_size, "bytes")



print("count ", count)
print("avg ", avgSize / count)
print("min ", minSize)
print("max ", maxSize)






