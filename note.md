### Tensorflow skillpath
- Resources:
    - Book: Deep Learning with Python, François Chollet
    - Documentation: Keras Library
    - Documentation: Tensorflow Library
    - Book: Algorithms of Oppression: How Search Engines Reinforce Racism, Safiya Umoja Noble
    - Book: Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy, Cathy O’Neil
- what is "deep"? : The deep part of deep learning refers to the numerous “layers” that transform data. This architecture mimics the structure of the brain, where each successive layer attempts to learn progressively complex patterns from the data fed into the model.
- There was a final step in the Perceptron algorithm that would give rise to the incredibly mysterious world of Neural Networks — the artificial neuron could train itself based on its own results, and fire better results in the future. In other words, it could learn by trial and error, just like a biological neuron.
- It was found out that creating multiple layers of neurons — with one layer feeding its output to the next layer as input — could process a wide range of inputs, make complex decisions, and still produce meaningful results. With some tweaks, the algorithm became known as the Multilayer Perceptron, which led to the rise of Feedforward Neural Networks.
- The data structure we use in deep learning is called a **tensor**, which is a generalized form of a vector and matrix: a multidimensional array.
- Neural network has different types of layers:
    - The value for each of these features would be held in an input node.
    - Hidden layers are layers that come between the input layer and the output layer. 
    - Output layer is the final layer in our neural network.
- The weighted sum between nodes and weights is calculated between each layer:
    - weighted_sum=(inputs⋅weight_transpose)+bias
    - Activation(weighted_sum)
    - Activation functions introduce nonlinearity in our learning model, creating more complexity during the learning process.
    - An activation function decides what is fired to the next neuron based on its calculation for the weighted sums.
- Forward propagation: input -> hidden note -> output
    - A bias node shifts the activation function either left or right to create the best fit for the given data in a deep learning model.
- Loss functions:
    - Cross-entropy loss: is used for classification learning models rather than regression.
- Backpropagration: computation of gradients with an algorithm known as gradient descent. This algorithm continuously updates and refines the weights between neurons to minimize our loss function.
- Gradient descent: parameter_new=parameter_old+learning_rate⋅gradient(loss_function(parameter_old))
- Stochastic Gradient Descent (SGD): Instead of performing gradient descent on our entire big dataset, we pick out a random data point to use at each iteration.
- Adam optimization algorithm: an adaptive learning algorithm that finds individual learning rates for each parameter. Ability to have an adaptive learning rate has made it an ideal variant of gradient descent and is commonly used in deep learning models.
- Mini-batch gradient descent is similar to SGD except instead of iterating on one data point at a time, we iterate on small batches of fixed size. Ideal trade-off between GD and SGD. Since mini-batch does not depend on just one training sample, it has a much smoother curve and is less affected by outliers and noisy data making it a more optimal algorithm for gradient descent than SGD.
- [Loss vs epochs among GD-SGD-Adam.jpg]
- Key issues:
    - Machine learning algorithms can only be as good as the data it is trained on
    - Machine learning models do not understand the impact of a false negative vs. a false positive diagnostic (at least not like humans can)
    - For many of the clinicians and the patients, the models are a black box.

##### Perceptron
- Already taken in Basic of Machine Learning course

##### Getting started with TensorFlow

##### Implementing neural network
- One-hot encoding for categorical features: `features = pd.get_dummies(features)`
- Train/test split
- Standardize/normalize numerical features
```py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer([('normalize', Normalizer(), ['age', 'bmi', 'children'])], remainder='passthrough') #returns NumPy arrays
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#convert a NumPy array back into a pandas DataFrame
features_train_norm = pd.DataFrame(features_train_norm, columns = features_train.columns)
# Note that we fit the scaler to the training data only, and then we apply the trained scaler onto the test data. This way we avoid “information leakage” from the training set to the test set.
```
- tf.keras.Sequential:
    - A sequential model, as the name suggests, allows us to create models layer-by-layer in a step-by-step fashion. This model can have only one input tensor and only one output tensor:
    ```py
    from tensorflow.keras.models import Sequential
    my_model = Sequential(name="my first model")
    my_model = design_model(features_train)  #refer to def design_model in script.py
    print(my_model.layers)  #The model’s layers are accessed via the layers attribute
    ```
    - Layers:
    ```py
    from tensorflow.keras import layers
    # we chose 3 neurons here
    layer = layers.Dense(3) 
    ```
    - Input layer:
    ```py
    from tensorflow.keras.layers import InputLayer
    my_input = InputLayer(input_shape=(15,)) #vinitializes an input layer for a DataFrame my_data that has 15 columns
    
    # OR avoiding hard-coding
    num_features = my_data.shape[1] 
    my_input = tf.keras.layers.InputLayer   (input_shape=(num_features,)) 

    # Add input layer to model
    my_model.add(my_input)
    print(my_model.summary())
    ```
    - Output layer
    ```py
    from tensorflow.keras.layers import Dense
    my_model.add(Dense(1))
    ```
- our model currently represents a linear regression. To capture more complex or non-linear interactions among the inputs and outputs neural networks, we’ll need to incorporate hidden layers.
```py
from tensorflow.keras.layers import Dense
my_model.add(Dense(64, activation='relu'))
# We chose 64 (2^6) to be the number of neurons since it makes optimization more efficient due to the binary nature of computation.
```
- There are a number of activation functions such as softmax, sigmoid, but ReLU (relu) (Rectified Linear Unit) is very effective in many applications and we’ll use it here.
- Adding more layers to a neural network naturally increases the number of parameters to be tuned.
- Keras offers a variety of optimizers such as SGD (Stochastic Gradient Descent optimizer), Adam, RMSprop.
- While model parameters are the ones that the model uses to make predictions, hyperparameters determine the learning process (learning rate, number of iterations, optimizer type).
```py
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.01)
```
- a model instance my_model is compiled:
```py
my_model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
# loss denotes the measure of learning success and the lower the loss the better the performance. 
# we want to observe the progress of the Mean Absolute Error (mae) while training the model because MAE can give us a better idea than MSE on how far off we are from the true values in the units we are predicting
```
- Training model: `my_model.fit(my_data, my_labels, epochs=50, batch_size=3, verbose=1)`
    - epochs refers to the number of cycles through the full training dataset. Since training of neural networks is an iterative process, you need multiple passes through data.
    - batch_size is the number of data points to work through before updating the model parameters. It is also a hyperparameter that can be tuned.
    - verbose = 1 will show you the progress bar of the training.
- Evaluating model: `val_mse, val_mae = my_model.evaluate(my_data, my_labels, verbose = 0)`
    - In our case, model.evaluate() returns the value for our chosen loss metrics (mse) and for an additional metrics (mae).

##### Hyperparameter tuning
- Hyperparameters are chosen on a held-out set called validation set
```py
my_model.fit(data, labels, epochs = 20, batch_size = 1, verbose = 1,  validation_split = 0.2)
```
- The batch size that determines how many training samples are seen before updating the network’s parameters (weight and bias matrices).
An advantage of using batches is for GPU computation that can parallelize neural network computations.
    - When the batch contains all the training examples, the process is called batch gradient descent. 
    - If the batch has one sample, it is called the stochastic gradient descent. 
    - When 1 < batch size < number of training points, is called mini-batch gradient descent. 
    - Probably bad performance for larger batch sizes. A good trick is to increase the learning rate!
- The number of epochs is a hyperparameter representing the number of complete passes through the training dataset. This is typically a large number (100, 1000, or larger). If the data is split into batches, in one epoch the optimizer will see all the batches.
    - Choose number of epochs: one trick is  early stopping: when the training performance reaches the plateau or starts degrading, the learning stops.
    ```py
    from tensorflow.keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size=16, verbose=0, validation_split=0.2, callbacks=[stop])
    #monitor = val_loss, which means we are monitoring the validation loss to decide when to stop the training
    #mode = min, which means we seek minimal loss
    #patience = 40, which means that if the learning reaches a plateau, it will continue for 40 more epochs in case the plateau leads to improved performance
    ```
- Adding hidden layer:
    - The validation curve is below the training curve. This means that the training curve can get better at some point, but the model complexity doesn’t allow it. This phenomenon is called underfitting. Can also notice that no early stopping occurs.
    - The rule of thumb is to start with one hidden layer and add as many units as we have features in the dataset. However, this might not always work. We need to try things out and observe our learning curve.

- Towards automated tuning: grid and random search. Random Search goes through random combinations of hyperparameters and doesn’t try them all.
    - Grid search in Keras: first wrap the neural network model in to a `KerasRegressor`:
    ```py
    model = KerasRegressor(build_fn=design_model)
    batch_size = [10, 40]
    epochs = [10, 50]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    # Initialize a GridSearchCV object and fit the model
    grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False))
    grid_result = grid.fit(features_train, labels_train, verbose = 0)
    # we initialized the scoring parameter with scikit-learn’s .make_scorer() method. We’re evaluating our hyperparameter combinations with a mean squared error making sure that greater_is_better is set to False since we are searching for a set of hyperparameters that yield us the smallest error.
    ```
    - Randomized search in Keras:  change our hyperparameter grid specification for the randomized search in order to have more options:
    ```py
    param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
    # Randomized search will sample values for batch_size and nb_epoch from uniform distributions in the interval [2, 16] and [10, 100], respectively, for a fixed number of iterations. In our case, 12 iterations:
    grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)
    ```
    - Regularization is a set of techniques that prevent the learning process to completely fit the model to the training data which can lead to overfitting. It makes the model simpler, smooths out the learning curve, and hence makes it more ‘regular’.
        - The most common regularization method is dropout: randomly ignores, or “drops out” a number of outputs of a layer by setting them to zeros
- A baseline result is the simplest possible prediction. 
```py
# Scikit-learn provides DummyRegressor, which serves as a baseline regression algorithm. We’ll choose mean (average) as our central tendency measure.
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features_train, labels_train)
y_pred = dummy_regr.predict(features_test)
MAE_baseline = mean_absolute_error(labels_test, y_pred)
print(MAE_baseline)
```

##### Classification
- Multi-class classification has the same idea behind binary classification, except instead of two possible outcomes, there are three or more.
- In multi-label classification, there are multiple possible labels for each outcome. This is useful for customer segmentation, image categorization, and sentiment analysis for understanding text. To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, SVMs, as well as various deep learning models.
- Cross-entropy is a score that summarizes the average difference between the actual and predicted probability distributions for all classes. 
    - log_loss() function in scikit-learn
- Datatrain.info()
- The following commands show us which categories we have in the item column and what their distribution is:
```py
from collections import Counter
print('Classes and number of values in the dataset`,Counter(data_train[“item”]))
#{‘lamps’: 75, ‘tableware’: 125, 'containers': 100}
```
- convert the label vectors to integers ranging from 0 to the number of classes by using sklearn.preprocessing.LabelEncoder:
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_y=le.fit_transform(train_y.astype(str))
test_y=le.transform(test_y.astype(str))
# We can print the resulting mappings
integer_mapping = {l: i for i, l in enumerate(le.classes_)}
print(integer_mapping)
# Convert to categorical format
train_y = tensorflow.keras.utils.to_categorical(train_y, dtype = ‘int64’)
test_y = tensorflow.keras.utils.to_categorical(test_y, dtype = ‘int64’)  
```
- For regression, we don’t use any activation function in the final layer because we needed to predict a number without any transformations. However, for classification:
    - vector of categorical probabilities: softmax activation function that outputs a vector with elements having values between 0 and 1 and that sum to 1 
    - binary classification problem: a sigmoid activation function paired with the binary_crossentropy loss.
- Setting the optimizer:
    - First, to specify the use of cross-entropy when optimizing the model, we need to set the loss parameter to categorical_crossentropy of the Model.compile() method.
    - Second, we also need to decide which metrics to use to evaluate our model. For classification, we usually use accuracy.
    - Finally, we will use Adam as our optimizer because it’s effective here and is commonly used.
- Train and evaluate models:
    - We take two outputs out of the .evaluate() function:
        - the value of the loss (categorical_crossentropy)
        - accuracy (as set in the metrics parameter of .compile()).
```py
my_model.fit(my_data, my_labels, epochs=10, batch_size=1, verbose=1)
loss, acc = my_model.evaluate(my_test, test_labels, verbose=0)
```
- Sometimes having only accuracy reported is not enough or adequate. Accuracy is often used when data is balanced, meaning it contains an equal or almost equal number of samples from all the classes. However, oftentimes data comes imbalanced, Eg. rare disease.
- Frequently, especially in medicine, false negatives and false positives have different consequences.
- F1-score is a helpful way to evaluate our model based on how badly it makes false negative mistakes:
```py
import numpy as np
from sklearn.metrics import classification_report
yhat_classes = np.argmax(my_model.predict(my_test), axis = 1)   #or axis=-1?
y_true = np.argmax(my_test_labels, axis=1)
print(classification_report(y_true, yhat_classes))
# predict classes for all test cases my_test using the .predict() method and assign the result to the yhat_classes variable.
# using .argmax() convert the one-hot-encoded labels my_test_labels into the index of the class the sample belongs to. The index corresponds to our class encoded as an integer.
# use the .classification_report() method to calculate all the metrics.
```
- There is another type of loss – sparse categorical cross-entropy – which is a computationally modified categorical cross-entropy loss that allows you to leave the integer labels as they are and skip the entire procedure of encoding.
    - save time in memory as well as computation because it uses a single integer for a class, rather than a whole vector. This is especially useful when we have data with many classes to predict.
    ```py
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # make sure that our labels are just integer encoded using the LabelEncoder() but not converted into one-hot-encodings using .to_categorical()
    ```
- Tweak the model: see if it can be improved
    - The first thing we can try is to increase the number of epochs.
    - Other hyperparameters you might consider changing are: the batch size number of hidden layers number of units per hidden layer the learning rate of the optimizer the optimizer and so on.

##### Image classification
- Use ImageDataGenerators to load images from a file path, and to preprocess them: `my_image_data_generator = ImageDataGenerator()`
- There are a few ways to preprocess image data, but we will focus on the most important step: pixel normalization. Because neural networks struggle with large integer values, we want to rescale our raw pixel values between 0 and 1. Our pixels have values in [0,255], so we can normalize pixels by dividing each pixel by 255.0.
- Can also use ImageDataGenerator for data augmentation: generating more data without collecting any new images. Common way is to randomly flip or shift each image by small amounts: `my_augmented_image_data_generator = ImageDataGenerator( vertical_flip = True )`
- Loading image data: using .flow_from_directory() method. Arguments:
    - directory : A string that defines the path to the folder containing our training data.
    - class_mode : How we should represent the labels in our data. “For example, we can set this to "categorical" to return our labels as one-hot arrays, with a 1 in the correct class slot.
    - color_mode : Specifies the type of image. For example, we set this to "grayscale" for black and white images, or to "rgb" (Red-Green-Blue) for color images.
    - target_size : A tuple specifying the height and width of our image. Every image in the directory will be resized into this shape.
    - batch_size : The batch size of our data.
- Once we have used these parameters to create our DirectoryIterator, we can iterate over the training batches using the .next() method.
```py
sample_batch_input,sample_batch_labels  = training_iterator.next()
print(sample_batch_input.shape,sample_batch_labels.shape)
```
- Feed-forward classification model to classify image: One way is to treat an image as a vector of pixels:
    - Change the shape of input layer to accept image data: (image height, image width, image channels): `model.add(tf.keras.Input(shape=(512,512,3)))`
    - "Flatten" input image into a single vector: `model.add(tf.keras.layers.Flatten())`
    - When we have flattened our image, we end up with an input of size 65536 features. We then need a matrix of size [65536 by 100] to transform this input into a layer with 100 units! This feed-forward model will struggle to learn meaningful combinations of so many features for those next hidden units.
- convolutional layers: allow us to scale down our input image into meaningful features while using only a fraction of the parameters required in a linear layer.
- Convolutional Neural Networks (CNNs) use layers specifically designed for image data. These layers capture local relationships between nearby features in an image.
    - we learn a set of smaller weight tensors, called filters (also known as kernels). We move each of these filters (i.e. convolve them) across the height and width of our input, to generate a new “image” of features. Each new “pixel” results from applying the filter to that location in the original image.
    - Convolution can reduce the size of an input image using only a few parameters.
    - Filters compute new features by only combining features that are near each other in the image. This operation encourages the model to look for local patterns (e.g., edges and objects).
    - Convolutional layers will produce similar outputs even when the objects in an image are translated (For example, if there were a giraffe in the bottom or top of the frame). This is because the same filters are applied across the entire image
    - With deep nets, we can learn each weight in our filter (along with a bias term)! Typically, we randomly initialize our filters and use gradient descent to learn a better set of weights. By randomly initializing our filters, we ensure that different filters learn different types of information, like vertical versus horizontal edge detectors.
- Configuring filters for convolutional layer
    - Define a Conv2D layer to handle the forward and backward passes of convolution: 
    ```py
    #Defines a convolutional layer with 4 filters, each of size 5 by 5:
    tf.keras.layers.Conv2D(4, 5, activation="relu"))
    ```
    - Then we stack these outputs of multiple filters together in a new “image.”  
    - Our output tensor is then (batch_size, new height, new width, number of filters). We call this last dimension number of channels ( or feature maps ). 
    - Filter size: [height,width,input channels], Keras takes care of the last dimension for us.
    - The number of parameters in a convolution layer: number of filters*(Input channels*height*Width + 1)  [Every filter has height, width, and thickness (The number of input channels), along with a bias term.]
- Configuring stride and padding for convolutional layers:
    - The stride hyperparameter is how much we move the filter each time we apply it: `model.add(tf.keras.layers.Conv2D(8, 5 strides=3, activation="relu"))`
    - The padding hyperparameter defines what we do once our filter gets to the end of a row/column.
        - The default option is to just stop when our kernel moves off the image. 
        - Another option is to pad our input by surrounding our input with zeros. This approach is called “same” padding, because if stride=1, the output of the layer will be the same height and width as the input: `model.add(tf.keras.layers.Conv2D(8, 5,strides=3,padding='same',activation="relu"))`
- Adding convolutional layers to model
    - Adding one convolutional layer: replace the first Dense layers with a Conv2D layer. Then, we want to move the Flatten layer between the convolutional and last dense  layer. 
    - Stacking Convolutional Layers (with distinct filter shapes and strides)
    ```py
    # 8 5x5 filters, with strides of 3
    model.add(tf.keras.layers.Conv2D(8, 5,  strides=3, activation="relu"))
    
    # 4 3x3 filters, with strides of 3
    model.add(tf.keras.layers.Conv2D(4, 3,  strides=3, activation="relu"))
    
    # 2 2x2 filters, with strides of 2
    model.add(tf.keras.layers.Conv2D(2, 3,  strides=2, activation="relu"))
    # The number of filters used in the previous layer becomes the number of channels that we input into the next!
    ```
    - As with dense layers, we should use non-linear activation functions between these convolutional layers.
- Another part of Convolutional Networks is Pooling Layers: layers that pool local information to reduce the dimensionality of intermediate convolutional outputs.
    - Most common type is Max pooling: instead of multiplying each image patch by a filter, we replace the patch with its maximum value: `max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),   strides=(3, 3), padding='valid')`
    - max pooling layers have another useful property: they provide some amount of translational invariance. In other words, even if we move around objects in the input image, the output will be the same. 
- Training the model:
    - Define another ImageDataGenerator and use it to load our validation data.
    - Compile our model with an optimizer, metric, and a loss function.
    - Train our model using model.fit().

##### Application of deep learning
- Recurrent Neural Network (RNN): Rather than just concatenating our input words together, RNNs process them sequentially. For every input word in order, we pass that word to our model. At each timestep, the input is then used to update the model’s hidden state. If we just want to predict the last word, we simply feed all but the last word into the model, then use the final hidden state to predict the next word. Alternatively, if we want to generate the entire sentence, we can feed a starting word in, update the hidden state, generate a new word, then pass that back into the new model, and so on.
- autoencoders:
    - The Encoder encodes input, and compresses it into a smaller latent representation, referred to as the Code.
    - The Decoder tries to reconstruct the input.
    - Our loss is the difference between our output and the original input. This is called the reconstruction loss.
- Autoencoders have many uses. For one, they can be used as a preprocessing step, to compress documents and images. These smaller vectors can also be used in downstream tasks like classification, clustering, or information retrieval.
- Without any additional labeled data, autoencoders can also be used for anomaly detection: the identification of rare, or suspicious data points (e.g. fake documents or credit card fraud).
- Generative Adversarial Networks (GANs).