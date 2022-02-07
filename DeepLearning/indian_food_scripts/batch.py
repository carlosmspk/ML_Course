from tensorflow.keras.utils import Sequence
import numpy as np
from matplotlib.image import imread, imsave
from os import listdir, mkdir
from tensorflow.keras.utils import to_categorical

def save_images_to_one_file(dataset_path, all_subpath = "all"):
    image_count = 0
    label_count = 0

    for dirname in listdir(dataset_path):
        if dirname in (".DS_Store", all_subpath):
            continue
        label_count += 1
        for image_path in listdir(dataset_path + "/" + dirname):
            image_count += 1

    labels = np.zeros((image_count,1))
    label = 0
    i = 0
    all_path = dataset_path + "/" + all_subpath + "/"

    for dirname in listdir(dataset_path):
        if dirname in (".DS_Store", all_subpath):
            continue
        for image_path in listdir(dataset_path + "/" + dirname):
            print (i, end="\r")
            from_path = dataset_path + "/" + dirname + "/" + image_path
            to_path = all_path + f"{i}.png"

            try:
                imsave(to_path, imread(from_path))
            except FileNotFoundError:
                mkdir(dataset_path + "/" + all_subpath)
                imsave(to_path, imread(from_path))
            labels[i] = label
            i += 1
        label += 1

    np.save(f"{dataset_path}/labels_one_hot.npy", to_categorical(labels))

    print (f"Saved {i} files, of {label} different labels. Images stored in {all_path}.")


class BatchGenerator(Sequence) :
  
  def __init__(self, image_names, images_path : str, labels : np.ndarray, batch_size) :
    self.images_path = images_path
    if not self.images_path.endswith("/"):
        self.images_path += "/"
    self.image_names = image_names
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(self.labels.shape[0] / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_names[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            imread(self.images_path + str(file_name), (300, 300, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)


            
if __name__ == "__main__":
    print(np.load("DeepLearning/dataset/IndianFood/labels_one_hot.npy").shape)