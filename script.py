### Implementing neural network
import pandas as pd

#load the dataset
dataset = pd.read_csv('insurance.csv') 
#choose first 7 columns as features
features = dataset.iloc[:,0:6] 
#choose the final column for prediction
labels = dataset.iloc[:,-1] 

#print the number of features in the dataset
print("Number of features: ", features.shape[1]) 
#print the number of samples in the dataset
print("Number of samples: ", labels.shape[0]) 
#see useful summary statistics for numeric features
print(features.describe()) 
print(labels.describe()) 

## Data preprocessing: one-hot encoding and standardization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#load the dataset
dataset = pd.read_csv('insurance.csv') 
#choose first 7 columns as features
features = dataset.iloc[:,0:6] 
#choose the final column for prediction
labels = dataset.iloc[:,-1] 

#one-hot encoding for categorical variables
features = pd.get_dummies(features) 
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
 
#normalize the numeric columns using ColumnTransformer
ct = ColumnTransformer([('normalize', Normalizer(), ['age', 'bmi', 'children'])], remainder='passthrough')
#fit the normalizer to the training data and convert from numpy arrays to pandas frame
features_train_norm = ct.fit_transform(features_train) 
#applied the trained normalizer on the test data and convert from numpy arrays to pandas frame
features_test_norm = ct.transform(features_test) 

#ColumnTransformer returns numpy arrays. Convert the features to dataframes
features_train_norm = pd.DataFrame(features_train_norm, columns = features_train.columns)
features_test_norm = pd.DataFrame(features_test_norm, columns = features_test.columns)

my_ct = ColumnTransformer([('scale', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train_scale = my_ct.fit_transform(features_train)
features_test_scale = my_ct.transform(features_test)
features_train_scale = pd.DataFrame(features_train_scale, columns = features_train.columns)
features_test_scale = pd.DataFrame(features_test_scale, columns = features_test.columns)
print(features_train_scale.describe())
print(features_test_scale.describe())

## tf.keras.Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def design_model(features):
  model = Sequential()
  return model
  
dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)

#print the layers
print(model.layers)

## Neural network model: layers
import tensorflow as tf
from tensorflow.keras import layers
layer = layers.Dense(3) #3 is the number we chose

print(layer.weights) #we get empty weight and bias arrays because tensorflow doesn't know what the shape is of the input to this layer

# 1338 samples, 11 features as in our dataset
input = tf.ones((1338, 11)) 
# a fully-connected layer with 3 neurons
layer = layers.Dense(3) 
# calculate the outputs
output = layer(input) 
# print the weights
print(layer.weights) 

## Neural network model: input, output, and hidden layer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense


def design_model(features):
  model = Sequential(name = "my_first_model")
  input = InputLayer(input_shape=(features.shape[1],))
  #add the input layer
  model.add(input) 
  #add the hidden layer
  model.add(Dense(128, activation='relu'))
  #adding an output layer to our model
  model.add(Dense(1)) 
  return model


dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)

#print the model summary here
print(model.summary())

## Optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def design_model(features):
  model = Sequential(name = "my_first_model")
  input = InputLayer(input_shape=(features.shape[1],))
   #add an input layer
  model.add(input)
  #add a hidden layer with 128 neurons
  model.add(Dense(128, activation='relu')) 
  #add an output layer
  model.add(Dense(1)) 
  #optimizer and model compile
  opt = Adam(learning_rate=0.01)
  model.compile(loss='mse', metrics=['mae'], optimizer=opt)
  return model

dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())

## Training and evaluating the model
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

tensorflow.random.set_seed(35) #for the reproducibility of results

def design_model(features):
  model = Sequential(name = "my_first_model")
  #without hard-coding
  input = InputLayer(input_shape=(features.shape[1],)) 
  #add the input layer
  model.add(input) 
  #add a hidden layer with 128 neurons
  model.add(Dense(128, activation='relu')) 
  #add an output layer to our model
  model.add(Dense(1)) 
  opt = Adam(learning_rate=0.1)
  model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
  return model

dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())

#fit the model using 40 epochs and batch size 1
model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)  #verbose set to True
#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test,labels_test, verbose =0)

### Hyperparameter tuning
#see model.py file for more details
from model import design_model, features_train, labels_train 

model = design_model(features_train, learning_rate = 0.01)
model.fit(features_train, labels_train, epochs= 40, batch_size=8, verbose=1, validation_split=0.33)

## Plotting loss on training/validation set with different learning rates
#see model.py file for more details
from model import design_model, features_train, labels_train 
import matplotlib.pyplot as plt

def fit_model(f_train, l_train, learning_rate, num_epochs, bs):
    #build the model
    model = design_model(f_train, learning_rate)
    #train the model on the training data
    history = model.fit(f_train, l_train, epochs = num_epochs, batch_size = bs, verbose = 0, validation_split = 0.2)
    # plot learning curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('lrate=' + str(learning_rate))
    plt.legend(loc="upper right")


#make a list of learning rates to try out
learning_rates = [1E-3, 1E-4, 1E-7]   #original [1, 0.1, 1E-7]
#fixed number of epochs
num_epochs = 100
#fixed number of batches
batch_size = 10 

for i in range(len(learning_rates)):
  plot_no = 420 + (i+1)
  plt.subplot(plot_no)
  fit_model(features_train, labels_train, learning_rates[i], num_epochs, batch_size)

plt.tight_layout()
plt.show()
plt.savefig('static/images/my_plot.png')
print("See the plot on the right with learning rates", learning_rates)
import app #don't worry about this. This is to show you the plot in the browser.

## Batch size
from model import features_train, labels_train, design_model
import matplotlib.pyplot as plt

def fit_model(f_train, l_train, learning_rate, num_epochs, batch_size, ax):
    model = design_model(features_train, learning_rate)
    #train the model on the training data
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size = batch_size, verbose=0, validation_split = 0.3)
    # plot learning curves
    ax.plot(history.history['mae'], label='train')
    ax.plot(history.history['val_mae'], label='validation')
    ax.set_title('batch = ' + str(batch_size), fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('# epochs')
    ax.set_ylabel('mae')
    ax.legend()

#In the previous checkpoint, you might have noticed bad performance for larger batch sizes (32 and 64). When having performance issues with larger batches it might help to increase the learning rate. Modify the value for the learning rate by assigning 0.1 to the learning_rate variable. 
learning_rate = 0.1  #original 0.01
#fixed number of epochs
num_epochs = 100
#we choose a number of batch sizes to try out
batches = [4, 32, 64]   #original [2, 10, 16] 
print("Learning rate fixed to:", learning_rate)

#plotting code
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.7, 'wspace': 0.4}) #preparing axes for plotting
axes = [ax1, ax2, ax3]

#iterate through all the batch values
for i in range(len(batches)):
  fit_model(features_train, labels_train, learning_rate, num_epochs, batches[i], axes[i])

plt.savefig('static/images/my_plot.png')
print("See the plot on the right with batch sizes", batches)
import app #don't worry about this. This is to show you the plot in the browser.

## Epochs and early stopping
from model import features_train, labels_train, design_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def fit_model(f_train, l_train, learning_rate, num_epochs):
    #build the model: to see the specs go to model.pyl we increased the number of hidden neurons
    #in order to introduce some overfitting
    model = design_model(features_train, learning_rate) 
    #train the model on the training data
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 16, verbose=0, validation_split = 0.2, callbacks = [es])
    return history

    
#using the early stopping in fit_model
learning_rate = 0.1
num_epochs = 500
history = fit_model(features_train, labels_train, learning_rate, num_epochs)

#plotting
fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.5}) 
(ax1, ax2) = axs
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='validation')
ax1.set_title('lrate=' + str(learning_rate))
ax1.legend(loc="upper right")
ax1.set_xlabel("# of epochs")
ax1.set_ylabel("loss (mse)")

ax2.plot(history.history['mae'], label='train')
ax2.plot(history.history['val_mae'], label='validation')
ax2.set_title('lrate=' + str(learning_rate))
ax2.legend(loc="upper right")
ax2.set_xlabel("# of epochs")
ax2.set_ylabel("MAE")

print("Final training MAE:", history.history['mae'][-1])
print("Final validation MAE:", history.history['val_mae'][-1])

plt.savefig('static/images/my_plot.png')
import app #don't worry about this. This is to show you the plot in the browser.

## Changing the model: adding the hidden layer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from model import features_train, labels_train

def more_complex_model(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(8, activation='relu'))  #original 64
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def one_layer_model(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def fit_model(model, f_train, l_train, learning_rate, num_epochs):
    #train the model on the training data
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 2, verbose=0, validation_split = 0.2, callbacks = [es])
    return history

def plot(history):
    # plot learning curves
    fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.8}) 
    (ax1, ax2) = axs
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('lrate=' + str(learning_rate))
    ax1.legend(loc="upper right")
    ax1.set_xlabel("# of epochs")
    ax1.set_ylabel("loss (mse)")

    ax2.plot(history.history['mae'], label='train')
    ax2.plot(history.history['val_mae'], label='validation')
    ax2.set_title('lrate=' + str(learning_rate))
    ax2.legend(loc="upper right")
    ax2.set_xlabel("# of epochs")
    ax2.set_ylabel("MAE")
    print("Final training MAE:", history.history['mae'][-1])
    print("Final validation MAE:", history.history['val_mae'][-1])

learning_rate = 0.1
num_epochs = 200

#fit the more simple model
print("Results of a one layer model:")
history1 = fit_model(one_layer_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history1)
plt.savefig('static/images/my_plot1.png')

#fit the more complex model
print("Results of a model with hidden layers:")
history2 = fit_model(more_complex_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history2)
plt.savefig('static/images/my_plot2.png')

import app #don't worry about this. This is to show you the plot in the browser.

## Towards automated tuning: grid and random search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from model import design_model, features_train, labels_train

#------------- GRID SEARCH --------------
def do_grid_search():
  batch_size = [6, 64]   #original [10, 40]
  epochs = [10, 50]
  model = KerasRegressor(build_fn=design_model)
  param_grid = dict(batch_size=batch_size, epochs=epochs)
  grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False),return_train_score = True)
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

  print("Training")
  means = grid_result.cv_results_['mean_train_score']
  stds = grid_result.cv_results_['std_train_score']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

#------------- RANDOMIZED SEARCH --------------
def do_randomized_search():
  param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
  model = KerasRegressor(build_fn=design_model)
  grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

print("-------------- GRID SEARCH --------------------")
do_grid_search()
print("-------------- RANDOMIZED SEARCH --------------------")
do_randomized_search()

## Regularization: dropout
from model import features_train, labels_train, fit_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from plotting import plot
import matplotlib.pyplot as plt

def design_model_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.3))


    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def design_model_no_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = layers.InputLayer(input_shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model
    
#using the early stopping in fit_model
learning_rate = 0.001
num_epochs = 200
#train the model without dropout
history1 = fit_model(design_model_no_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
#train the model with dropout
history2 = fit_model(design_model_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)

plot(history1, 'static/images/no_dropout.png')

plot(history2, 'static/images/with_dropout.png')

import app #don't worry about this. This is to show you the plot in the browser.

## Baselines
#see model.py file for more details
from model import features_train, labels_train, features_test, labels_test
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


dummy_regr = DummyRegressor(strategy="median")  #original mean
dummy_regr.fit(features_train, labels_train)
y_pred = dummy_regr.predict(features_test)
MAE_baseline = mean_absolute_error(labels_test, y_pred)
print(MAE_baseline)

