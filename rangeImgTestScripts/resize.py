


from PIL import Image
from PIL import ImageOps
import numpy as np

path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"



image = Image.open(path)
imageGrey = ImageOps.grayscale(image)


newSize = (512, 32)
# newSize = (256, 16)

imageGrey = imageGrey.resize(newSize)

imageGrey.show()

imageArray = np.array(imageGrey)

imageGreyArrayNorm = imageArray.astype('float32') / 255
image_arr = np.expand_dims(imageGreyArrayNorm, axis=2)

print(np.shape(image_arr))






