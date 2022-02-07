from tensorflow.keras.utils import Sequence
import numpy as np
from matplotlib.image import imread
from os import listdir
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from pickle import dump, load

def save_img_data_to_pickle(dataset_path):
    try:
        with open(dataset_path + "/daa.pkl", "rb") as f:
            data = load(f)
            print ("Dataset already batched. Returning data.")
        return data
    except FileNotFoundError:
        pass

    IGNORE_FILES = (".DS_Store", "data.pkl")

    image_count = 0
    file_names = []

    for dirname in listdir(dataset_path):
        if dirname in IGNORE_FILES:
            continue
        for image_path in listdir(dataset_path + "/" + dirname):
            image_count += 1
            file_names.append(image_path)

    i = 0
    labels = np.zeros((image_count, 1))
    label = 0
    for dirname in listdir(dataset_path):
        if dirname in IGNORE_FILES:
            continue
        for image_path in listdir(dataset_path + "/" + dirname):
            labels[i] = label
            i += 1
        label += 1
            
    
    data = {
        "labels": to_categorical(labels),
        "img_paths": file_names
        }
    with open(dataset_path + "/data.pkl", "wb") as f:
        dump(data, f)
    


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
    save_img_data_to_pickle("DeepLearning/dataset/IndianFood")