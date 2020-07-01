# About Data
The Fashion MNIST data set contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels).

# Data Preprocessing

When using a convolutional layer as the first layer to our model, we need to reshape our data to (n_images, x_shape, y_shape, channels). All I really need to know is that I should set channels to 1 for grayscale images and set channels to 3 when you have a set of RGB-images as input.
here in this Sequential neural network i'm using grey scale images so channel is equals to 1.

# Training
This convolutional layers will have 128 neurons (feature maps) and a 3x3 feature detector. In turn, our pooling layers will use max pooling with a 2x2 matrix. Convolutional neural networks are almost always proceeded by an artificial neural network. In Keras, a Dense layer implements the operation output = activation(dot(input, weight) + bias). The input to our artificial neural network must be in one dimension therefore we flatten it beforehand.


