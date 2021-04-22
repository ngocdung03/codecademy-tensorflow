##Implementing Neural Networks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  #can also import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('life_expectancy.csv')
print(dataset.head())
print(dataset.describe())
# Drop 'Country' because to create a predictive model, knowing from which country data comes can be confusing and it is not a column we can generalize over.
dataset = dataset.drop(['Country'], axis=1)

labels = dataset.iloc[:,-1]
features = dataset.iloc[:,0:-1]

# apply one-hot-encoding on all the categorical columns
dataset = pd.get_dummies(dataset)

# split the data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state=1)

# Standardize/normalize
numerical_features = features_train.select_dtypes(include=['float64', 'int64']) #only select numerical features types automatically
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_features.columns)], remainder='passthrough')  
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)
# le=LabelEncoder()
# labels_train=le.fit_transform(labels_train.astype(str))
# labels_test=le.transform(labels_test.astype(str))

# Building the model
my_model = Sequential()
input = InputLayer(input_shape=(features_train.shape[1],))
my_model.add(input)
my_model.add(Dense(64, activation='relu'))
my_model.add(Dense(1))
print(my_model.summary())
# optimizer
opt = Adam(learning_rate =0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
#train
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)
#print(labels_train.unique())
res_mse, res_mae = my_model.evualuate(features_test_scaled, labels_test)
print(res_mse)
print(res_mae)