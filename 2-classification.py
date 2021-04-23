### Classification
# dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
## Classification
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv('heart_failure.csv')
print(data.info())
# Distribution of death_event
print(Counter(data['death_event']))
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
# One-hot ecoding to convert from categorical features into vectors
x = pd.get_dummies(x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1)

ct = ColumnTransformer([('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])], remainder='passthrough')  #why not 'standardize'?
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Label encoding for categorical outcome
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))
# Convert labels into categorical type
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Model
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax')) # number of neurons in output layer corresponding to the number of classes in the dataset.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=16)
loss, acc = model.evaluate(X_test, Y_test)
y_estimate = model.predict(X_test)
# select the indices of the true classes for each label encoding in y_estimate
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Print additional metrics, such as F1-score
print(classification_report(y_true, y_estimate))