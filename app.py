import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import matplotlib.pyplot as plt
import numpy as np
from iteration.py import newFilter

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class names
class_names = ['Tom', 'Jerry', 'Both', 'Neither']

# Build CNN model
array = [
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        # layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
        ]

while True:
    model = keras.Sequential(array)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    filter = newFilter()
    index = len(array) - 2
    array.insert(index, filter)


    break


# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
