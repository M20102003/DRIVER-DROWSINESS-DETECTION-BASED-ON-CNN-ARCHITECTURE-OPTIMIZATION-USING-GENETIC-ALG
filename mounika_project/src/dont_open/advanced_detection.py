import os
import time
import cv2
import numpy as np
import pygame
from threading import Thread

# Install required packages if not already installed
try:
    import dlib
except ImportError:
    os.system("pip install dlib")
    import dlib

try:
    import pygame
except ImportError:
    os.system("pip install pygame")
    import pygame

# Initialize pygame mixer for alarm sound
pygame.mixer.init()

# Constants
EYE_CLOSED_THRESHOLD = 0.25  # Eye aspect ratio threshold
CLOSED_EYES_DURATION = 2.0  # Time in seconds for alarm to trigger
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALARM_VOLUME = 0.8  # Volume level (0.0 to 1.0)

# Function to create an alarm sound
def generate_beep_sound():
    alarm_file = "alarm_sound.wav"
    if not os.path.exists(alarm_file):
        # Create a simple beep sound using pygame
        sample_rate = 44100
        duration = 1.0  # 1 second
        beep_freq = 880  # A5 note (Hz)
        
        # Generate beep waveform
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(beep_freq * 2 * np.pi * t) * 32767 * 0.5
        beep = beep.astype(np.int16)
        
        # Save as WAV file
        from scipy.io.wavfile import write
        write(alarm_file, sample_rate, beep)
    
    return alarm_file

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye_points):
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alarm sound in a separate thread
def play_alarm(sound_file):
    def _play_sound():
        sound = pygame.mixer.Sound(sound_file)
        sound.set_volume(ALARM_VOLUME)
        sound.play()
        time.sleep(1)  # Let it play for 1 second
        pygame.mixer.stop()
    
    # Start in a separate thread to avoid blocking
    Thread(target=_play_sound).start()

# Main function
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    
    # Download the shape predictor if it doesn't exist
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("Downloading facial landmark predictor...")
        import urllib.request
        url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, compressed_file)
        
        # Extract the file
        import bz2
        with open(predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_file, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)
        
        # Remove the compressed file
        os.remove(compressed_file)
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Generate alarm sound
    alarm_sound = generate_beep_sound()
    
    # Variables for tracking eye closure
    eyes_closed_start_time = None
    alarm_active = False
    continuous_detection_frames = 0
    
    # Define indices for left and right eyes (based on 68-point facial landmarks)
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))
    
    print("Starting eye closure detection. Press 'q' to quit.")
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # Status text to display
        status_text = "Status: No face detected"
        status_color = (0, 0, 255)  # Red by default
        
        # Process each detected face
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
            # Extract eye coordinates
            left_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in left_eye_indices])
            right_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in right_eye_indices])
            
            # Draw eye contours
            cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
            
            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            
            # Average the eye aspect ratio of both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), FONT, 0.7, (0, 255, 0), 2)
            
            # Check if eyes are closed (EAR below threshold)
            if ear < EYE_CLOSED_THRESHOLD:
                continuous_detection_frames += 1
                
                # Start timer if eyes just closed
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                    
                # Calculate elapsed time with eyes closed
                elapsed_time = time.time() - eyes_closed_start_time
                
                # Display closed time
                status_text = f"Eyes CLOSED for {elapsed_time:.1f}s"
                status_color = (0, 0, 255)  # Red
                
                # Trigger alarm if eyes closed for longer than threshold
                if elapsed_time >= CLOSED_EYES_DURATION and not alarm_active:
                    alarm_active = True
                    play_alarm(alarm_sound)
                    
                    # Make the alert more visible
                    cv2.putText(frame, "WAKE UP!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                FONT, 1.2, (0, 0, 255), 3)
            else:
                # Reset timer and status if eyes are open
                continuous_detection_frames = 0
                eyes_closed_start_time = None
                alarm_active = False
                status_text = "Eyes OPEN"
                status_color = (0, 255, 0)  # Green
            
            # Reset alarm state if eyes stay open for a while
            if continuous_detection_frames == 0:
                alarm_active = False
        
        # Display status
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), FONT, 0.7, status_color, 2)
        
        # Display the frame
        cv2.imshow("Eye Closure Detection", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()