import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras_preprocessing.image import load_img, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.notebook import tqdm

# Use absolute paths
BASE_DIR = 'C:/Users/Aryan/Desktop/mini project'
TRAIN_DIR = os.path.join(BASE_DIR, 'images/train')
TEST_DIR = os.path.join(BASE_DIR, 'images/test')

def createDataFrame(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_dir = os.path.join(dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            for imagename in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, imagename))
                labels.append(label)
            print(label, "COMPLETED")
    return image_paths, labels

# Create DataFrames for training and testing data
train = pd.DataFrame()
train['image'], train['label'] = createDataFrame(TRAIN_DIR)
test = pd.DataFrame()
test['image'], test['label'] = createDataFrame(TEST_DIR)

# Function to extract features from images
def extractFeatures(images):
    features = []
    for image in tqdm(images, desc="Extracting features"):
        img = load_img(image, color_mode="grayscale", target_size=(48, 48))  # Updated
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Extract features for training and testing data
train_features = extractFeatures(train['image'])
test_features = extractFeatures(test['image'])
x_train = train_features / 255.0
x_test = test_features / 255.0

# Encode labels
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Build the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                    epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Define the directory to save the model
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Print the paths to check
print("Saving model architecture to:", os.path.join(MODEL_DIR, "FaceEmotionRecognitionNew.json"))
print("Saving model weights to:", os.path.join(MODEL_DIR, "FaceEmotionRecognitionNew.weights.h5"))

# Save the model to JSON
model_json = model.to_json()
with open(os.path.join(MODEL_DIR, "FaceEmotionRecognitionNew.json"), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(MODEL_DIR, "FaceEmotionRecognitionNew.weights.h5"))

print("Model saved successfully.")
