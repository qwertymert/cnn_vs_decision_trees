from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_images_from_folder(folder, img_size=128):
    images_train = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    images_train_label = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    images_val = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    images_val_label = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    images_test = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    images_test_label = {"COVID19": [], "PNEUMONIA": [], "NORMAL": [], "TUBERCULOSIS": []}
    folders = ["test", "train", "val"]
    for folder_name in folders:
        if folder_name == "test":
            data_dict = images_test
            label_dict = images_test_label
        elif folder_name == "train":
            data_dict = images_train
            label_dict = images_train_label
        else:
            data_dict = images_val
            label_dict = images_val_label
        folder_path = os.path.join(folder, folder_name)
        for disease_folder in os.listdir(folder_path):
            if disease_folder == "NORMAL":
                label = 0
            elif disease_folder == "COVID19":
                label = 1
            elif disease_folder == "PNEUMONIA":
                label = 2
            else:
                label = 3
            disease_path = os.path.join(folder_path, disease_folder)
            idx = 0
            for img_name in os.listdir(os.path.join(folder_path, disease_folder)):
                img_path = os.path.join(disease_path, img_name)
                img = Image.open(img_path)
                img = img.resize((img_size, img_size)).convert("L")
                data_dict[disease_folder].append(np.array(img))
                label_dict[disease_folder].append(label)
                idx += 1
                if idx % 100 == 0:
                    print(f"Loading {folder_name} {disease_folder} images : {idx}")
    return (images_train, images_train_label), (images_val, images_val_label), (images_test, images_test_label)


def scale_dataset(x, y, scale_method="standard", shuffle=True, seed=0, scaler=None):
    x_set = np.concatenate(list(x.values()))
    shape_x = x_set.shape
    x_set = x_set.reshape(shape_x[0], -1)

    if scaler is None:
        if scale_method == "minmax":
            scaler = MinMaxScaler()
            scaler = scaler.fit(x_set)
        else:
            scaler = StandardScaler()
            scaler = scaler.fit(x_set)

    x_set = scaler.transform(x_set)
    x_set = x_set.reshape(shape_x)
    x_set = np.expand_dims(x_set, axis=1)
    y_set = np.concatenate(list(y.values()))

    if shuffle:
        np.random.seed(seed)
        idx = np.random.permutation(len(x_set))
        x_set = x_set[idx]
        y_set = y_set[idx]

    return x_set, y_set, scaler
