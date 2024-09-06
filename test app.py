from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image  # Importing the missing Image class
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.CRITICAL)

# Define the folder for saving training images and loss plots
TRAINING_DATA_FOLDER = 'training_data'
PLOTS_FOLDER = 'plots'

# Ensure the folders exist
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load preprocessed Quick, Draw! data
X_train = np.load('quickdraw_data/X_train.npy')
y_train = np.load('quickdraw_data/y_train.npy')

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Prepare the data for PyTorch
X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train_encoded = torch.tensor(y_train_encoded, dtype=torch.long)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Create a dataset and dataloader
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_encoded)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, criterion, and optimizer
labels = label_encoder.classes_.tolist()
model = SimpleNN(len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def retrain_model():
    """Retrain the model with the current training data."""
    model.train()
    training_losses = []
    for epoch in range(500):
        epoch_loss = 150.0
        for batch_images, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_losses.append(epoch_loss / len(dataloader))
        logging.info(f"Epoch {epoch+1}, Loss: {training_losses[-1]}")
    
    # Plot and save the training loss graph
    plt.figure()
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'training_loss.png'))
    plt.close()

    logging.info("Model retraining complete")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((28, 28)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1).numpy().flatten()

    predictions = sorted(zip(labels, confidences), key=lambda x: x[1], reverse=True)[:10]

    return jsonify({'guesses': [{'label': label, 'confidence': conf * 100} for label, conf in predictions]})

@app.route('/train', methods=['POST'])
def train():
    global model, criterion, optimizer, labels

    data = request.json
    image_data = data['image']
    label = data['label'].capitalize()  # Capitalize the first letter
    width = data.get('width', 28)
    height = data.get('height', 28)

    logging.debug(f"Received label: {label}")

    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((width, height)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    label_folder = os.path.join(TRAINING_DATA_FOLDER, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        labels.append(label)
        labels.sort()
        model = SimpleNN(len(labels))  # Update model to reflect new number of classes
        criterion = nn.CrossEntropyLoss()  # Reinitialize criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reinitialize optimizer
    
    # Save the image with a unique name
    image_path = os.path.join(label_folder, f"{label}_{len(os.listdir(label_folder))}.png")
    image.save(image_path)
    
    label_index = labels.index(label)
    label_tensor = torch.tensor([label_index], dtype=torch.long)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(image_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
    
    logging.debug("Training complete")

    retrain_model()  # Retrain the model after adding a new image

    return jsonify({'status': 'training complete'})

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to manually trigger retraining."""
    retrain_model()
    return jsonify({'status': 'retraining complete'})

@app.route('/plot')
def plot():
    """Endpoint to serve the training loss plot."""
    plot_path = os.path.join(PLOTS_FOLDER, 'training_loss.png')
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
