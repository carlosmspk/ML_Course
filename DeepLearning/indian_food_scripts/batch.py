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
            print("Dataset already batched. Returning data.")
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
    one_hot_shuffled = to_categorical(labels_shuffled)

    data = {"labels": one_hot_shuffled, "img_paths": file_names_shuffled}
    with open(dataset_path + "/data.pkl", "wb") as f:
        dump(data, f)


class BatchGenerator(Sequence):
    def __init__(self, dataset_path: str, image_paths, labels, batch_size):
        self.image_paths = image_paths
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        if not self.dataset_path.endswith("/"):
            self.dataset_path += "/"

    def __len__(self):
        return (np.ceil(self.labels.shape[0] / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        return np.array(
            [
                imread(self.dataset_path + image_path, (300, 300, 3))
                for image_path in batch_x
            ]
        ), np.array(batch_y)


if __name__ == "__main__":
    BATCH_SIZE = 100
    save_img_data_to_pickle("DeepLearning/dataset/IndianFood")
    with open("DeepLearning/dataset/IndianFood/data.pkl", "rb") as f:
        data = load(f)
    img_paths = data["img_paths"]
    labels = data["labels"]
    batch_gen = BatchGenerator(
        dataset_path="DeepLearning/dataset/IndianFood",
        image_paths=img_paths,
        labels=labels,
        batch_size=BATCH_SIZE,
    )
    analyzed_images = 0

    for i, (data_batch, label_batch) in enumerate(batch_gen):
        analyzed_images += len(label_batch)
        print(
            f"{i}: {data_batch.shape}\t{label_batch.shape}\tanalyzed {analyzed_images} out of {len(batch_gen.labels)}"
        )
