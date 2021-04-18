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
    