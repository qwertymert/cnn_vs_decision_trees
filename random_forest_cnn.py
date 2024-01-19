from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import numpy as np
import torch

from dataset_tools import load_images_from_folder, scale_dataset
from feature_extract import FeatureExtractor

seed = 0
torch.manual_seed(seed)

np.random.seed(seed)


image_size = 256
seed = 0

# load data
(x_train_dict, y_train_dict), (x_val_dict, y_val_dict), (x_test_dict, y_test_dict) = load_images_from_folder("main_dataset", image_size)

x_train, y_train, scaler = scale_dataset(x_train_dict, y_train_dict, "standard", True, seed)
x_val, y_val, _ = scale_dataset(x_val_dict, y_val_dict, "standard", True, seed, scaler)
x_test, y_test, _ = scale_dataset(x_test_dict, y_test_dict, "standard", False, seed, scaler)


x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

extractor = FeatureExtractor(pooling=True)


train_features = extractor.extract_mobilenet_feature(x_train)
val_features = extractor.extract_mobilenet_feature(x_val)
test_features = extractor.extract_mobilenet_feature(x_test)


feature_scaler = StandardScaler()
x_train_features = feature_scaler.fit_transform(np.array(train_features))
x_val_features = feature_scaler.transform(np.array(val_features))
x_test_features = feature_scaler.transform(test_features)


train_idx = np.arange(x_train_features.shape[0])
np.random.shuffle(train_idx)
x_train_features, y_train = x_train_features[train_idx], y_train[train_idx]

val_idx = np.arange(x_val_features.shape[0])
np.random.shuffle(val_idx)
x_val_features, y_val = x_val_features[val_idx], y_val[val_idx]

test_idx = np.arange(x_test_features.shape[0])
np.random.shuffle(test_idx)
x_test_features, y_test = x_test_features[test_idx], y_test[test_idx]


time1 = time.time()

model = RandomForestClassifier(n_estimators=75)
model.fit(x_train_features, y_train)

print(time.time()-time1)

pred = model.predict(x_test_features)
report = classification_report(y_test, pred)
print(report)
conf_matrix = confusion_matrix(y_test, pred)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
display.plot()
plt.show()
