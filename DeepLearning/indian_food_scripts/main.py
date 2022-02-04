from matplotlib import image as img
import numpy as np
from os import listdir, chdir

chdir("DeepLearning")
n_images = 0
n_labels = 0
encoder = dict()
for dirname in listdir("dataset/IndianFood"):
    if dirname == ".DS_Store":
        continue
    for image in listdir("dataset/IndianFood/" + dirname):
        n_images += 1
    encoder[n_labels] = dirname
    n_labels += 1
    

label = 0
x = np.zeros((n_images,300,300,3))
y = np.zeros((n_images, n_labels))

i = 0
label = 0

for dirname in listdir("dataset/IndianFood"):
    if dirname == ".DS_Store":
        continue
    for image in listdir("dataset/IndianFood/" + dirname):
        x[i] = img.imread("dataset/IndianFood/" + dirname + "/" + image)
        y[i][label] = 1

print (x.shape)
print (y.shape)