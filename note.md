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
