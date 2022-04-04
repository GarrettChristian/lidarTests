from PIL import Image
from PIL import ImageOps
import numpy as np

imgPath = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"

image = Image.open(imgPath)
image = ImageOps.grayscale(image)
image.show()

greyImg = np.array(image)
print(np.shape(greyImg))
flatImg = greyImg.flatten()
print(np.shape(flatImg))

