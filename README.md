# Data Mining and Convolutional Neural Networks for Chest Disease Prediction


# Setup

1. Install requirements with `pip install -r requirements.txt`

# Downloading the data

Download the data from [here](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis?select=val) and extract it to the `main_dataset` folder.

Directory structure should look like this:

```shell
Project
└─── main_dataset
     ├── train 
     │   
     ├── val
     │  
     └── test
```

# Training

To run and reproduce the decision tree and random forest experiment, use the command `python decision_trees.py`

To run and reproduce the Random Forest with CNN experiment, use the command `python random_forest_cnn.py`

To run and reproduce the CNN experiment, use the command `python cnn.py`


# Results
### Decision Tree model
- Hyperparameter settings

| Hyperparameter     | Value       |
|--------------------|-------------|
| image_size         | 128         |
| criterion          | gini        |
| min_samples_split  | 2           |
| min_samples_leaf   | 1           |

- Classification report
```shell
              precision    recall  f1-score   support

           0       0.68      0.32      0.44       234
           1       0.74      0.63      0.68       106
           2       0.70      0.91      0.79       390
           3       0.42      0.66      0.51        41

    accuracy                           0.68       771
   macro avg       0.64      0.63      0.61       771
weighted avg       0.69      0.68      0.65       771
```

- Confusion matrix

<p align="left">
<img src="images/output_dec.png" height = "300" alt="" align=center />
</p>

### Random Forest model
- Hyperparameter settings

| Hyperparameter   | Value |
|------------------|-------|
| image_size       | 128   |
| n_estimators     | 100   |
| criterion        | gini  |
|min_samples_split | 2     |
|min_samples_leaf  | 1     |

- Classification report
```shell
              precision    recall  f1-score   support

           0       0.95      0.33      0.49       234
           1       0.94      0.75      0.84       106
           2       0.72      0.98      0.83       390
           3       0.45      0.80      0.58        41

    accuracy                           0.74       771
   macro avg       0.77      0.72      0.69       771
weighted avg       0.81      0.74      0.72       771
```

- Confusion matrix

<p align="left">
<img src="images/output.png" height = "300" alt="" align=center />
</p>

### Random Forest with CNN extraction model

- Hyperparameter settings

| Hyperparameter   | Value       |
|------------------|-------------|
| image_size       | 256         |
| n_estimators     | 75          |
| criterion        | gini        |
|min_samples_split | 2           |
|min_samples_leaf  | 1           |
|feature_extractor | mobilenetv2 |

- Classification report
```shell
              precision    recall  f1-score   support

           0       0.94      0.47      0.63       234
           1       1.00      0.54      0.70       106
           2       0.69      0.98      0.81       390
           3       0.85      0.80      0.83        41

    accuracy                           0.76       771
   macro avg       0.87      0.70      0.74       771
weighted avg       0.82      0.76      0.74       771
```

- Confusion matrix

<p align="left">
<img src="images/random_forest_cnn.png" height = "300" alt="" align=center />
</p>
  


### CNN model

- Final test set accuracy : 0.7899
- Loss and accuracy curves

<p align="center">
<img src="images/resnet50_loss_curve.png" height = "300" alt="" align=center />
</p>

<p align="center">
<img src="images/resnet50_acc_curve.png" height = "300" alt="" align=center />
</p>
