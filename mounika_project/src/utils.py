import os
import cv2

# Install missing libraries
try:
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2

# Function to detect face
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return faces, gray

# Function to detect eyes
def detect_eyes(face_roi):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(face_roi)
    return eyes