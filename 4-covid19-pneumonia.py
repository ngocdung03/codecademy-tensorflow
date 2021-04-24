# data: refer to /src/DATA/classification-challenge.zip
# data: https://www.kaggle.com/pranavraikokte/covid19-image-dataset
## Covid-19 and Pneumonia Classification with Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

# Look at the dataset:
# train and test folders
# view the images: all are in grayscale
# there are 3 different folders within train/test, incidicating a multi-class classifcation problem rather than binary classfication.
# Load image data
data_generator = ImageDataGenerator(rescale=1.0/255)  # If you flip or crop your images during data augmentation, note that you would have to create separate ImageDataGenerator() objects for both for your training and validation data to avoid hurting the performance on your test set.

Class_mode = 'categorical'   # You can set your class_mode to sparse (instead of 'categorical')and use sparse categorical loss instead for this model 
Color_mode='grayscale'
Target_size= (256,256)
Batch_size=5   # You can play with the batch_size parameter as you fine-tune your model.
training_iterator = data_generator.flow_from_directory('augmented-data/train',class_mode=Class_mode,color_mode=Color_mode,target_size=Target_size,batch_size=Batch_size)  #from tensorflow.keras.preprocessing.image module

validation_iterator = data_generator.flow_from_directory('augmented-data/test',class_mode=Class_mode,color_mode=Color_mode,target_size=Target_size,batch_size=Batch_size)

print(training_iterator.image_shape)
print(validation_iterator.image_shape)
# Fit the model
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=20)
model = Sequential()
model.add(tf.keras.Input(shape=training_iterator.image_shape))
model.add(tf.keras.layers.Conv2D(2, (5, 5), strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(4, (3, 3), strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(tf.keras.layers.Dense(3,activation="softmax"))
model.summary()
model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
   loss=tf.keras.losses.CategoricalCrossentropy(),
   metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)
# Fit the model: Optional for Sequence: if steps_per_epoch unspecified, will use the len(generator) as a number of steps.

model.fit(training_iterator, 
#steps_per_epoch=x_train.shape[0]/Batch_size, 
epochs=1000, 
validation_data=validation_iterator,
#validation_steps=y_train.shape[0]/Batch_size,
callbacks = [es])

# Plot the cross-entropy loss for both the train and validation data over each epoch using the Matplotlib Library. You can also plot the AUC metric for both your train and validation data as well. This will give you an insight into how the model performs better over time and can also help you figure out better ways to tune your hyperparameters.
#Because of the way Matplotlib plots are displayed in the learning environment, please use fig.savefig('static/images/my_plots.png') at the end of your graphing code to render the plot in the browser. If you wish to display multiple plots, you can use .subplot() or .add_subplot() methods in the Matplotlib library to depict multiple plots in one figure.
# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
 
fig.savefig('static/images/my_plots.png')
# use this savefig call at the end of your graph instead of using plt.show()
plt.savefig('static/images/my_plots.png')

#Another potential extension to this project would be implementing a classification report and a confusion matrix. These are not tools we have introduced you to; however, if you would like further resources to improve your neural network, we recommend looking into these concepts.
#As a brief introduction, these concepts evaluate the nature of false positives and false negatives in your neural network taking steps beyond simple evaluation metrics like accuracy.
#In the hint below, you will see a possible solution to calculate a classification_report and a confusion_matrix, but you will need to do some personal googling/exploring to acquaint yourself with these metrics and understand the outputs.
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)
print(cm)