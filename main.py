# Import required libraries
import csv
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from matplotlib.cbook import flatten

from tensorflow.keras import models, datasets, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense


# Initialize data containers and label names
image_list_train = []
label_list_train = []
image_list_test = []
label_list_test = []
label_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load training data from csv
with open('C:/Users/fatem/Downloads/Fashion MNist/fashion-mnist_train.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    #reader = csv.DictReader(csvfile)
    #header = reader.fieldnames

# print(header)

    for row in csvreader:
        label = int(row[0])
        pixels = np.array(row[1:], dtype=np.uint8). reshape(28,28)/255.0

        image_list_train.append(pixels)
        label_list_train.append(label)

# Total number of images and their shape
print(len(image_list_train), pixels.shape)
#print(pixels.shape)
#print(image_list_train[0])

# Plot 25 sample training images
plt.figure(figsize=(10,10))
for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_list_train[i])
        plt.xlabel(label_names[label])
plt.show()

# Load training data from csv
with open('C:/Users/fatem/Downloads/Fashion MNist/fashion-mnist_test.csv') as csvfile:
    csvreader2 = csv.reader(csvfile)
    next(csvreader2)

    for row in csvreader2:
        label=int(row[0])
        pixels = np.array(row[1:], dtype=np.uint8). reshape(28,28)/255.0

        image_list_test.append(pixels)
        label_list_test.append(label)

#print(len(image_list_test), pixels.shape)
#print(image_list_test[0])

# Plot 25 sample testing images
plt.figure(figsize=(10,10))
for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_list_test[i])
        plt.xlabel(label_names[label])
plt.show()

# Should be <class 'keras.layers. Flatten'>
print(type(Flatten))

# Define the model architecture
def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=image_list_train[0].shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

model = get_model()

model.summary()
model.layers[1].get_weights()

# Print layer weights and biases
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w, b = weights
        print(f"\nLayer {i} ({layer.name}):")
        print("Weights shape:", w.shape)
        print("Biases shape:", b.shape)


# Convert lists to NumPy arrays
image_list_train = np.array(image_list_train)
label_list_train = np.array(label_list_train)
image_list_test = np.array(image_list_test)
label_list_test = np.array(label_list_test)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(image_list_train, label_list_train, epochs=10)

# Evaluate model performance on test data
print("Starting evaluation...")
results = model.evaluate(image_list_test, label_list_test)
print("Evaluation done.")

# Visualize predictions for 10 test images (1 from each class)
plt.figure(figsize=(10,10))
for i in range(10):
        plt.subplot(5, 2 ,i+1)
        test = image_list_test[label_list_test == i][0]
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test.reshape(28,28))

        # Predict the label using the model
        pred = model.predict(test.reshape(1, 28, 28))
        idx = np.argmax(pred[0])
        pred_class = label_names[idx]

        # Show actual and predicted labels
        plt.xlabel('Original {} | Predict {}' .format(label_names[i], pred_class))
plt.tight_layout()
plt.show()




# print(label_list[0],label_names[label_list[0]])
# cv2.imshow('First image', image_list[0])
# cv2.waitKey(0)

#for i in range(len(image_list_train)):
#    cv2.imshow(label_names[i], image_list_train[i])
#    cv2.waitKey(0)
#pred = model.predict(test.reshape(1,28,28))
#idx = np.argmax(pred[0])
#pred_class = label_names[idx]
#print(pred)
#print(idx)
#print(pred_class)

# Print results with labels
#print(f"Test Loss: {results[0]:.4f}")
#print(f"Test Accuracy: {results[1]*100:.2f}%")



