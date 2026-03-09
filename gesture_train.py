import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "D:/hand_gesture_CNN/LeapGestRecog"

IMG_SIZE = 64

data = []
labels = []

print("Loading dataset...")

# Loop through persons (00,01,02...)
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    # Loop through gesture folders
    for gesture_folder in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        # Extract label from folder name (01_palm -> 0)
        label = int(gesture_folder.split('_')[0]) - 1

        for img_name in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

print("Images loaded:", len(data))

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Convert labels to categorical
labels = to_categorical(labels, num_classes=10)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("gesture_model.h5")

print("Model saved as gesture_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()