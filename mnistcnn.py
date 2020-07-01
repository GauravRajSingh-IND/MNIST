# Importing required Libraries
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Loading Data as cvs files
# Here two different dataset is used for train and testing
train = pd.read_csv("/content/drive/My Drive/Dog_Cat/mnist_train.csv")
test = pd.read_csv("/content/drive/My Drive/Dog_Cat/mnist_test.csv")

# Understanding the Data
# Shape of datasets
print("Training dataset shape: ",train.shape)
print("Testing dataset shape:  ",test.shape)

train.head()

# Seprating features and labels for traing data
# x_train contains all the features
# y_train contains all the labels
x_train = train.drop("label", axis = 1)
y_train = train["label"]
print(x_train.head(5))
print(y_train.head(2))

# Seprating features and labels for testing data
# x_test contains all the features
# y_test contains all the labels
x_test = test.drop("label", axis = 1)
y_test = test["label"]
print(x_test.head(5))
print(y_test.head(2))

# Data Preprocessing
# Converting x_train, x_test data to array
x_train = np.asarray(x_train)
x_test = np.array(x_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Reshaping the data to (28,28) shape image
# 1 represents channel of the image

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


print("Training dataset shape:       ",x_train.shape)
print("Testing dataset shape:        ",x_test.shape)
print("Training label dataset shape: ",y_train.shape)
print("Testing label dataset shape:  ",y_test.shape)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalization of image data
x_train = x_train / 255.0
x_test = x_test / 255.0

img_path = x_train[30]
plt.imshow(np.squeeze(img_path))

# Simple Sequential network
# softmax activation is used in output layer ( classiffication problem)

model = Sequential()
model.add(Conv2D(128, kernel_size= 3, activation= 'relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D())
model.add(Conv2D(64, kernel_size= 3, activation= 'relu'))
model.add(Dropout(0.20))
model.add(Conv2D(32, kernel_size= 3, activation= 'relu'))
model.add(Flatten())
model.add(Dense(10, activation= 'softmax'))

model.summary()

model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(x_train, y_train, epochs= 10, batch_size= 128)
model.evaluate(x_test, y_test)


