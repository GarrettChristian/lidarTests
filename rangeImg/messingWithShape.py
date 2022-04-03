from lib2to3.pgen2.token import EQUAL
from PIL import Image
from PIL import ImageOps
import numpy as np

# imgPath = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"
# imgPath = '/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/rangeImg/rangeimgs/00/001372.png'
imgPath = "000000.png"
# imgPath = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/rangeImg/unrealistic1.png"

image = Image.open(imgPath)
X = np.array(image)
print(np.shape(X))

# print(imgPath)
# im = Image.fromarray(X)
# im.save("your_file.png")


imageGrey = ImageOps.grayscale(image)
grey = np.array(imageGrey)
print(np.shape(grey))
print(grey)
# image.show()

newShape = np.expand_dims(grey, axis=2)
print(np.shape(newShape))
print(newShape)

undoExpandDims = newShape.reshape(64, 1024)
print(np.shape(undoExpandDims))

decodedImage = Image.fromarray(undoExpandDims)
decodedImage.show()

