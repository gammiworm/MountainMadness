import os
import shutil
import random

# Define dataset directories
train_dir = os.path.join("tom_and_jerry_training_dataset")
test_dir = os.path.join("tom_and_jerry_testing_dataset")

# Create test directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

# Define categories
categories = ["tom", "jerry", "both", "neither"]

for category in categories:
    train_category_path = os.path.join(train_dir, category)
    test_category_path = os.path.join(test_dir, category)
    
    # Ensure test subdirectories exist
    os.makedirs(test_category_path, exist_ok=True)
    
    # List all images in training category
    images = os.listdir(train_category_path)
    random.shuffle(images)
    
    # Determine number of images to move (20%)
    num_test_images = int(len(images) * 0.2)
    test_images = images[:num_test_images]
    
    # Move images to test directory and remove from train
    for img in test_images:
        src_path = os.path.join(train_category_path, img)
        dest_path = os.path.join(test_category_path, img)
        shutil.move(src_path, dest_path)
    
    print(f"Moved {num_test_images} images from {category} to test directory and removed them from train directory.")

print("Dataset successfully split into train and test directories.")
