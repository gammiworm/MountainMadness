import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import numpy as np
import os

dataset_path = "/app/tom_and_jerry_training_dataset"
dataset_testing_path = "/app/tom_and_jerry_testing_dataset"


# Load dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(64, 64),  # Resize images to a reasonable size
    batch_size= 64, # Since we only have 5000 images, we can afford a small batch size
    label_mode="int"  # Can be "categorical" or None if you don’t have labels
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

# train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# Define train-validation split
train_size = 0.8  # 80% training, 20% validation
val_size = 1 - train_size

# Get total number of batches
total_batches = len(train_dataset)

# Calculate number of training batches
train_batches = int(total_batches * train_size)

# Split dataset
#train_dataset = train_dataset.take(train_batches)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(64, 64),
    batch_size=64,
    validation_split=0.2,  # Use 20% for validation
    subset="training",
    seed=123  # Ensures consistent split
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(64, 64),
    batch_size=64,
    validation_split=0.2,  # Use same split
    subset="validation",
    seed=123
)
#val_dataset = train_dataset.skip(train_batches)




# Define class names
class_names = ['tom', 'jerry', 'both', 'neither']

# Build CNN model

model = keras.Sequential([
        keras.Input(shape=(64, 64, 3)),  # Define input explicitly
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),  # Reduce size

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),  # Reduce size
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=2)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_acc:.2f}")

model.save("tom_and_jerry_classifier.h5")

def predict_single_image(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get index of max probability

    class_names = ["Tom", "Jerry", "Both", "Neither"]  # Update with your class names
    return class_names[predicted_class]  # Return the predicted class label

def predict_images_in_directory(directory_path, model):
    class_names = ["Tom", "Jerry", "Both", "Neither"]  # Update class labels
    predictions = {}  # Store predictions for each image

    # Loop through all image files in the directory
    for subdirectory in os.listdir(directory_path):
        for filename in subdirectory:
            file_path = os.path.join(directory_path,subdirectory,filename)

            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize

                # Predict class
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)  # Get index of highest probability

                # Store result
                predictions[filename] = class_names[predicted_class]

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    return predictions  # Return all results as a dictionary

model = keras.models.load_model("tom_and_jerry_classifier.h5")
print("Test image prediction: " + predict_single_image("test_image.jpg", model))

results = predict_images_in_directory(dataset_testing_path, model)

for img_name, predicted_label in results.items():
    print(f"{img_name}: {predicted_label}")



