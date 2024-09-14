import cv2
from keras.models import model_from_json
import numpy as np
import os

# Define file paths
json_file_path = "C:\\Users\\Aryan\\Desktop\\mini project\\FaceEmotionRecognitionNew.json"
weights_file_path = "C:\\Users\\Aryan\\Desktop\\mini project\\FaceEmotionRecognitionNew.weights.h5"
haar_file_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load the model architecture
if os.path.exists(json_file_path):
    print(f"Loading model architecture from: {json_file_path}")
    json_file = open(json_file_path, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
else:
    print(f"Error: JSON file not found at {json_file_path}")
    exit(1)

# Load the model weights
if os.path.exists(weights_file_path):
    print(f"Loading model weights from: {weights_file_path}")
    model.load_weights(weights_file_path)
else:
    print(f"Error: Model weights file not found at {weights_file_path}")
    exit(1)

# Load Haar Cascade file
if os.path.exists(haar_file_path):
    print(f"Loading Haar Cascade file from: {haar_file_path}")
    face_cascade = cv2.CascadeClassifier(haar_file_path)
else:
    print(f"Error: Haar Cascade file not found at {haar_file_path}")
    exit(1)

def extractFeatures(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extractFeatures(image)
            pred = model.predict(img)
            prediction_label = labels[np.argmax(pred)]
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        cv2.imshow("OUTPUT", im)
        cv2.waitKey(1)

    except cv2.error:
        pass
