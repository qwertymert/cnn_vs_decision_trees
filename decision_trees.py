from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import numpy as np

from dataset_tools import load_images_from_folder

seed = 0
np.random.seed(seed)

new_size = 128
train_set, val_set, test_set = load_images_from_folder("main_dataset", new_size)

scaler = StandardScaler()

x_train = scaler.fit_transform(np.concatenate(list(train_set[0].values())).reshape(-1, new_size**2))
y_train = np.concatenate(list(train_set[1].values()))
train_idx = np.arange(x_train.shape[0])
np.random.shuffle(train_idx)
x_train, y_train = x_train[train_idx], y_train[train_idx]


x_val = scaler.transform(np.concatenate(list(val_set[0].values())).reshape(-1, new_size**2))
y_val = np.concatenate(list(val_set[1].values()))
val_idx = np.arange(x_val.shape[0])
np.random.shuffle(val_idx)
x_val, y_val = x_val[val_idx], y_val[val_idx]

x_test = scaler.transform(np.concatenate(list(test_set[0].values())).reshape(-1, new_size**2))
y_test = np.concatenate(list(test_set[1].values()))
test_idx = np.arange(x_test.shape[0])
np.random.shuffle(test_idx)
x_test, y_test = x_test[test_idx], y_test[test_idx]


total = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
print(total)
print("train size", x_train.shape[0])
print("test size", x_test.shape[0])
print("val size", x_val.shape[0])
print("train-test ratio :", x_test.shape[0]/x_train.shape[0])


time1 = time.time()
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(time.time()-time1)


pred = model.predict(x_test)
report = classification_report(y_test, pred)
print(report)
conf_matrix = confusion_matrix(y_test, pred)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
display.plot()
plt.show()


time2 = time.time()
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
print("Total training time :", time.time()-time2)
pred = forest.predict(x_test)
report = classification_report(y_test, pred)
print(report)


conf_matrix = confusion_matrix(y_test, pred)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=forest.classes_)
display.plot()
plt.show()
