import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Load the data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Filter out invalid labels (labels >= 43)
train_filter = y_train < 43
X_train = X_train[train_filter]
y_train = y_train[train_filter]

val_filter = y_val < 43
X_val = X_val[val_filter]
y_val = y_val[val_filter]

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=64)

# Save the model
model.save("traffic_model.h5")
print("Model saved as traffic_model.h5")
