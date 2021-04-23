### Classifying Galaxies
# data: https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

# Because the dataset comprises over one thousand images, you’ll use a custom function, load_galaxy_data() to load the compressed data files into the Codecademy learning environment as NumPy arrays.
input_data, labels = load_galaxy_data()
print(input_data.shape)  # (1400, 128, 128, 3)  128px tall x wide, 3 channels RGB
print(labels.shape)

x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=222)  #stratify argument ensures the ratios of galaxies will be the same between the data

# Preprocess the input
data_generator = ImageDataGenerator(rescale=1.0/255)  #perform pixel normalization 

# Create two NumpyArrayIterators
Batch_size=5
training_iterator = data_generator.flow(x_train, y_train, batch_size=Batch_size)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=Batch_size)

# Build the model
model = Sequential()
model.add(Input(shape=(128, 128, 3))) #refer to input_data.shape
# Adding two convolutional layers, interspersed with max pooling layers, followed by two dense layers:
model.add(Conv2D(8, 3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(8, 3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
# Output
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy(),AUC()])  # Because the labels are one-hot categories, use tf.keras.losses.CategoricalCrossentropy() as your loss. 
print(model.summary())

# Train
model.fit(training_iterator, steps_per_epoch=x_train.shape[0]/Batch_size, epochs=8, validation_data=validation_iterator, validation_steps=y_train.shape[0]/Batch_size)  #can be len(x_train) instead

# model’s accuracy should be around 0.60-0.70, and your AUC should fall into the 0.80-0.90 range: val_categorical_accuracy: 0.6152 - val_auc: 0.8441
# Your accuracy tells you that your model assigns the highest probability to the correct class more than 60% of the time. For a classification task with over four classes, this is no small feat: a random baseline model would achieve only ~25% accuracy on the dataset. Your AUC tells you that for a random galaxy, there is more than an 80% chance your model would assign a higher probability to a true class than to a false one.

#If you would like, try tweaking your architecture. Can you find a better set of hyperparameters? MAKE SURE to watch your parameter count: it’s easy to accidentally create a model with more than tens of thousands of parameters, which could overfit to your relatively small dataset (or crash the Learning Environment). Note that scores will fluctuate a bit, depending on how the weights are randomly initialized.
# learning rate
# number of convolutional layers
# number of filters, strides, and padding type per layer
# stride and pool_size of max pooling layers
# size of hidden linear layers

# Visualize how your convolutional neural network processes images
# These feature maps showcase the activations of each filter as they are convolved across the input.
from visualize import visualize_activations
visualize_activations(model, validation_iterator)
# visualize_results takes your Keras model and the validation iterator and does the following:
# It loads in a sample batch of data using your validation iterator.
# It uses model.predict() to generate predictions for the first sample images.
# Next, it compares those predictions with the true labels and prints the result.
# It then saves the image and the feature maps for each convolutional layer using matplotlib.
