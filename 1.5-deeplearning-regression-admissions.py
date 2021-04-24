## Deep Learning Regression with Admissions Data
# data: refer to /src/DATA/regression-challenge.zip
# data: https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv
import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
## Deep Learning Regression with Admissions Data
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score


dataset = pd.read_csv('admissions_data.csv')
print(dataset.describe())
# Luckily, there are no categorical variables in this dataset, so you do not have to perform one-hot encoding. You should always check this, however, before diving into analysis!
# dataset = pd.get_dummies(dataset)
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# Standardize
#numerical_features = features_train.select_dtypes(include=['float64', 'int64']) #only select numerical features types automatically
ct = ColumnTransformer([("only numeric", StandardScaler(), features.columns)], remainder='passthrough')  
x_train_scaled = ct.fit_transform(x_train)
x_test_scaled = ct.transform(x_test)

# Do extensions code below
my_model = Sequential()
my_model.add(InputLayer(input_shape=(x_train_scaled.shape[1],)))
my_model.add(Dense(16, activation='relu'))
my_model.add(Dense(1))
print(my_model.summary())
# optimizer
opt = Adam(learning_rate =0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
#train
history = my_model.fit(x_train_scaled, y_train, epochs=40, batch_size=1, verbose=1, validation_split = 0.3)   #without validation_split, cannot extract val_mae
my_model.evaluate(x_test_scaled, x_test)

# PLOT the model loss per epoch as well as the mean-average error per epoch for both training and validation data. This will give you an insight into how the model performs better over time and can also help you figure out better ways to tune your hyperparameters.
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('static/images/my_plots.png')

# Let’s say you wanted to evaluate how strongly the features in admissions.csv predict an applicant’s admission into a graduate program. We can use something called an R-squared value.
predicted_values = my_model.predict(x_test_scaled) 
print(r2_score(y_test, predicted_values)) 