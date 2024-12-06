import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Dropout
from sklearn.model_selection import train_test_split
import os
import cv2

# Load the pre-trained ResNet50 model without the top layer
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from a video
def extract_features(video_path):
    features = []
    for frame in os.listdir(video_path):
        frame_path = os.path.join(video_path, frame)
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        img = tf.keras.applications.resnet50.preprocess_input(img)  # Preprocess for ResNet
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        feature = resnet_model.predict(img)
        features.append(feature)
    return np.array(features)

# Load the dataset
train_dir = 'new_dataset/train'
test_dir = 'new_dataset/test'

train_videos = os.listdir(train_dir)
test_videos = os.listdir(test_dir)

train_features = []
train_labels = []
test_features = []
test_labels = []

for video in train_videos:
    video_path = os.path.join(train_dir, video)
    features = extract_features(video_path)
    train_features.append(features)
    # Assuming the label is the same as the video name
    train_labels.append(video)

for video in test_videos:
    video_path = os.path.join(test_dir, video)
    features = extract_features(video_path)
    test_features.append(features)
    # Assuming the label is the same as the video name
    test_labels.append(video)

# Convert to numpy array
train_features = np.array(train_features)
test_features = np.array(test_features)

# Reshape features for LSTM input
time_steps = 10  # Number of frames to consider for each sample
num_features = train_features.shape[2]  # Number of features from ResNet
num_samples = train_features.shape[1] // time_steps

# Prepare the data for LSTM
X_train = train_features[:, :num_samples * time_steps].reshape(-1, time_steps, num_features)
y_train = np.array(train_labels)[:num_samples]  # Adjust labels accordingly

X_test = test_features[:, :num_samples * time_steps].reshape(-1, time_steps, num_features)
y_test = np.array(test_labels)[:num_samples]  # Adjust labels accordingly

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(time_steps, num_features), return_sequences=False))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(len(train_videos), activation='softmax'))  # Change output layer based on your task

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')