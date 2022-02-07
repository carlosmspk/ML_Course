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
            file_names.append(f"{dirname}/{image_path}")

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
    
    file_names_shuffled, labels_shuffled = shuffle(file_names, labels)
            
    
    data = {
        "labels": to_categorical(labels_shuffled),
        "img_paths": file_names_shuffled
        }
    with open(dataset_path + "/data.pkl", "wb") as f:
        dump(data, f)
    


class BatchGenerator(Sequence) :
  
  def __init__(self, dataset_path : str, batch_size) :
    with open(dataset_path + "/data.pkl", "rb") as f:
        data = load(f)
    
    self.dataset_path = dataset_path
    if not dataset_path.endswith("/"):
        self.dataset_path += "/"
    self.image_paths, self.labels = data["img_paths"], data["labels"]
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(self.labels.shape[0] / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            imread(self.dataset_path + str(image_path), (300, 300, 3))
               for image_path in batch_x])/255.0, np.array(batch_y)


            
if __name__ == "__main__":
    save_img_data_to_pickle("DeepLearning/dataset/IndianFood")
    batch_gen = BatchGenerator("DeepLearning/dataset/IndianFood", batch_size=22)
    for data_batch, label_batch in batch_gen:
        print (data_batch.shape, label_batch.shape)
        exit()