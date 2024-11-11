# -*- coding: utf-8 -*-
"""Real-Time.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bcz8OzeuJEVPpUxDRMRekw9E4Ojo80MR
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Accessing your dataset (replace 'path_to_your_dataset' with the actual path)
dataset_path = '/content/drive/My Drive/Real-Time_Dataset'  # Adjust to your dataset's path

# Assuming the dataset is organized with a folder for each sign
#dataset_path = '/path/to/dataset'  # Replace with your dataset path
labels = ['Hello', 'Yes', 'No', 'ThankYou', 'ILikeYou']
label_map = {label: idx for idx, label in enumerate(labels)}

# Load and preprocess data
def load_data(dataset_path):
    images = []
    image_labels = []

    for label in labels:
        sign_folder = os.path.join(dataset_path, label)
        for image_file in os.listdir(sign_folder):
            image_path = os.path.join(sign_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Unable to read image at {image_path}")
                continue

            image = cv2.resize(image, (64, 64))
            images.append(image)
            image_labels.append(label_map[label])

    images = np.array(images, dtype='float32') / 255.0
    image_labels = to_categorical(np.array(image_labels), num_classes=len(labels))

    return train_test_split(images, image_labels, test_size=0.2, random_state=42)

# Load dataset
X_train, X_test, y_train, y_test = load_data(dataset_path)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
hist = model.fit(X_train, y_train, epochs=32, batch_size=16, validation_data=(X_test, y_test))

# Save the model
model.save('sign_language_model.h5')

from matplotlib import pyplot as plt

fig = plt.figure()

plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=14)
plt.legend(loc="upper left")

plt.show()

fig = plt.figure()

plt.plot(hist.history['loss'], color='green', label='loss')
plt.plot(hist.history['val_loss'],color='red',label='val_loss')
fig.suptitle('Loss',fontsize=14)
plt.legend(loc="upper left")

plt.show()

# Evaluate the model
predictions = model.predict(X_test)
y_pred = np.round(predictions)
y_true = y_test

from sklearn.metrics import classification_report, confusion_matrix

# Print classification report and confusion matrix
import numpy as np

# Convert one-hot encoded labels to single integer labels
y_true_single = np.argmax(y_true, axis=1)
y_pred_single = np.argmax(y_pred, axis=1)

# Now you can use classification_report and confusion_matrix
print("Classification Report:\n", classification_report(y_true_single, y_pred_single))
print("Confusion Matrix:\n", confusion_matrix(y_true_single, y_pred_single))

#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming y_true_single and y_pred_single are your true and predicted labels
cm = confusion_matrix(y_true_single, y_pred_single)

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()