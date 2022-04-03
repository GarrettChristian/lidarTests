from PIL import Image
from PIL import ImageOps
import numpy as np

imgPath = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/000000.png"

image = Image.open(imgPath)
image = ImageOps.grayscale(image)
image.show()

greyImg = np.array(image)
greyImgNorm = greyImg.astype('float32') / 255
greyImgNormUndo = greyImgNorm.astype('float32') * 255
print(greyImgNormUndo)
print(np.shape(greyImgNormUndo))


normImage = Image.fromarray(greyImgNormUndo)
normImage.show()

print("same ", np.array_equal(greyImgNormUndo, greyImg))


# imageGrey = ImageOps.grayscale(image)
# imageGreyArray = np.array(imageGrey)
# imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
# X = np.expand_dims(imageGreyArrayNorm, axis=2)

# print(X)
# print(np.shape(X))