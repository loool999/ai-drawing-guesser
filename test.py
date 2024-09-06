import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Define the categories and corresponding files
categories = {}
data_folder = 'quickdraw_data'

for file in os.listdir(data_folder):
    if file.endswith(".npy"):
        category = file.split('.')[0]
        categories[category] = os.path.join(data_folder, file)

# Function to preprocess the data
def preprocess_data(categories, img_size=28, save_interval=100000):
    images = []
    labels = []
    processed_count = 0
    for label, file in categories.items():
        try:
            data = np.load(file, mmap_mode='r')
            if data.size == 0:
                raise ValueError(f"File {file} is empty.")
            for img in data:
                if img.shape != (784,):
                    raise ValueError(f"Unexpected shape {img.shape} in file {file}.")
                img = img.reshape(28, 28)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
                if len(images) % 1000 == 0:  # Process in smaller batches
                    print(f"Processed {len(images)} images so far")
                if processed_count % save_interval == 0:  # Save every save_interval images
                    print(f"Saving data at {processed_count} images.")
                    X_train = np.array(images).reshape(-1, img_size, img_size, 1)
                    y_train = np.array(labels)
                    np.save('quickdraw_data/X_train.npy', X_train)
                    np.save('quickdraw_data/X_test.npy', X_test)
                    np.save('quickdraw_data/y_train.npy', y_train)
                    np.save('quickdraw_data/y_test.npy', y_test)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    images = np.array(images).reshape(-1, img_size, img_size, 1)
    labels = np.array(labels)
    return images, labels

# Preprocess and save the data
images, labels = preprocess_data(categories)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save the preprocessed data
np.save('quickdraw_data/X_train.npy', X_train)
np.save('quickdraw_data/X_test.npy', X_test)
np.save('quickdraw_data/y_train.npy', y_train)
np.save('quickdraw_data/y_test.npy', y_test)
