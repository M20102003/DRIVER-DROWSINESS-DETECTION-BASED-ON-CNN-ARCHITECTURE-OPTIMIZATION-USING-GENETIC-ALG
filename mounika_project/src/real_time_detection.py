import os
import time
import threading
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import mediapipe as mp
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DrowsinessDetector')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class DrowsinessDetector:
    def __init__(self):
        print("inside class")
        # Paths and directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.alarm_path = os.path.join(self.base_dir, "alarm.wav")
        
        # Detection parameters
        self.eye_closed_threshold = 0.25  # EAR threshold for closed eyes
        self.drowsiness_time_threshold = 2.0  # Seconds before alarm triggers
        self.consecutive_frames_threshold = 10  # Number of consecutive frames to confirm drowsiness
        self.eye_closed_duration = 0.0  # Track how long eyes have been closed
        self.last_eye_state = True  # True for open, False for closed
        self.closed_eye_start_time = None
        
        # State tracking
        self.alarm_active = False
        self.drowsy_count = 0
        self.awake_count = 0
        
        # Create a buffer to smooth detection results
        self.drowsiness_buffer = deque(maxlen=15)  # Store last 15 frame results
        
        self._initialize_model()
        self._initialize_face_detector()
        self._initialize_audio()
    
    def _initialize_model(self):
        try:
            # Configure TensorFlow for memory growth to avoid OOM errors
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Found GPU: {physical_devices}")
            
            # Load the pre-trained model
            model_path = os.path.join(self.models_dir, "cnn_model.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            self.model = load_model(model_path)
            self.model.summary()  # Print model architecture for debugging
            logger.info("CNN model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _initialize_face_detector(self):
        try:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Define the eye landmarks indices
            # Left eye landmarks indices
            self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
            # Right eye landmarks indices
            self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
            # Iris indices
            self.LEFT_IRIS = [474, 475, 476, 477]
            self.RIGHT_IRIS = [469, 470, 471, 472]
            
            logger.info("MediaPipe Face Mesh initialized")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise
    
    def _initialize_audio(self):
        try:
            # Initialize pygame for audio playback
            pygame.mixer.init()
            if os.path.exists(self.alarm_path):
                # Test load to ensure file works
                pygame.mixer.music.load(self.alarm_path)
                pygame.mixer.music.set_volume(0.7)  # Set volume to 70%
                logger.info("Audio system initialized")
            else:
                logger.warning(f"Alarm sound file not found: {self.alarm_path}")
                # Create a simple beep sound as fallback
                self._create_fallback_alarm()
        except Exception as e:
            logger.error(f"Error initializing audio: {e}")
            # Continue without audio rather than crashing
            self._create_fallback_alarm()
    
    def _create_fallback_alarm(self):
        """Create a simple beep sound as a fallback if the audio file is missing"""
        try:
            import numpy as np
            from scipy.io.wavfile import write
            
            # Generate a simple beep sound
            sample_rate = 44100
            t = np.linspace(0, 2, 2 * sample_rate, False)
            beep = np.sin(2 * np.pi * 440 * t) * 0.5
            
            # Save as a WAV file
            fallback_path = os.path.join(self.base_dir, "fallback_alarm.wav")
            write(fallback_path, sample_rate, beep.astype(np.float32))
            self.alarm_path = fallback_path
            logger.info(f"Created fallback alarm at {fallback_path}")
        except Exception as e:
            logger.error(f"Failed to create fallback alarm: {e}")
    
    def play_alarm(self):
        """Play the alarm sound in a separate thread"""
        if self.alarm_active:
            return  # Already playing
        
        self.alarm_active = True
        try:
            logger.info("Playing alarm")
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.play()
            # Keep track of whether the sound is still playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error playing alarm: {e}")
        finally:
            self.alarm_active = False
            logger.info("Alarm stopped")
    
    def calculate_ear(self, landmarks, eye_indices):
        """
        Calculate the Eye Aspect Ratio (EAR) using facial landmarks from MediaPipe
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            points = []
            for i in eye_indices:
                position = landmarks.landmark[i]
                points.append([position.x, position.y])
            
            points = np.array(points)
            
            # Compute the horizontal distance
            horizontal = np.linalg.norm(points[0] - points[3])
            
            # Compute the two vertical distances
            vertical1 = np.linalg.norm(points[1] - points[5])
            vertical2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate EAR
            ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)  # Add small constant to avoid division by zero
            
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.3  # Return a default value that won't trigger alarm
    
    def detect_blinks(self, ear, frame_number):
        """
        Detect blinks based on EAR value.
        Returns True if eyes are open, False if closed.
        """
        # Threshold for determining closed eyes
        if ear < self.eye_closed_threshold:
            self.drowsy_count += 1
            self.awake_count = 0
            return False
        else:
            self.awake_count += 1
            if self.awake_count > 3:  # Reset drowsy count if eyes open for 3+ frames
                self.drowsy_count = 0
            return True
    
    def check_drowsiness_with_model(self, frame):
        """Use CNN model to classify drowsiness"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using Haar Cascade for backup
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess for model input
                try:
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = face_roi / 255.0
                    face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
                    face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
                    
                    # Make prediction
                    prediction = self.model.predict(face_roi, verbose=0)
                    return prediction[0][0] < 0.5  # Return True if drowsy (closed eyes)
                except Exception as e:
                    logger.error(f"Error in model prediction: {e}")
            
            return False  # No drowsiness detected if no faces found
        except Exception as e:
            logger.error(f"Error in model-based drowsiness detection: {e}")
            return False
    
    def process_frame(self, frame, frame_number):
        """
        Process a single frame to detect drowsiness.
        Returns: (processed_frame, is_drowsy)
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return frame, False
            
            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Process the frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            # Draw the face mesh on the frame (for visualization)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear = self.calculate_ear(face_landmarks, self.LEFT_EYE_INDICES)
                    right_ear = self.calculate_ear(face_landmarks, self.RIGHT_EYE_INDICES)
                    
                    # Average EAR of both eyes
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Draw landmarks and eye lines for visualization
                    self._draw_landmarks(frame, face_landmarks, ear)
                    
                    # Detect eye state (open/closed)
                    eyes_closed = not self.detect_blinks(ear, frame_number)
                    
                    # Add to detection buffer for smoothing
                    self.drowsiness_buffer.append(eyes_closed)
                    
                    # Check for consistent drowsiness (eyes closed for several consecutive frames)
                    is_drowsy = False
                    current_time = time.time()
                    
                    # State transition: Eyes changed from open to closed
                    if not self.last_eye_state and eyes_closed:
                        # Eyes are still closed
                        pass
                    elif self.last_eye_state and eyes_closed:
                        # Eyes just closed - start timer
                        if self.closed_eye_start_time is None:
                            self.closed_eye_start_time = current_time
                    elif not self.last_eye_state and not eyes_closed:
                        # Eyes just opened - reset timer
                        self.closed_eye_start_time = None
                    
                    # Check if eyes have been closed long enough to be considered drowsy
                    if self.closed_eye_start_time is not None:
                        self.eye_closed_duration = current_time - self.closed_eye_start_time
                        if self.eye_closed_duration >= self.drowsiness_time_threshold:
                            is_drowsy = True
                            
                            # Verify drowsiness with CNN for extra confirmation
                            model_drowsy = self.check_drowsiness_with_model(frame)
                            if model_drowsy:
                                is_drowsy = True
                                # Reset for next detection
                                self.closed_eye_start_time = None
                    
                    self.last_eye_state = eyes_closed
                    
                    # Display drowsiness status
                    self._display_status(frame, is_drowsy, ear)
                    
                    return frame, is_drowsy
            
            # If face mesh fails, try backup method with the CNN model
            is_drowsy = self.check_drowsiness_with_model(frame)
            
            # Display basic status
            cv2.putText(frame, f"Drowsy: {is_drowsy}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)
            
            return frame, is_drowsy
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, False
    
    def _draw_landmarks(self, frame, face_landmarks, ear):
        """Draw facial landmarks and eye measurements on the frame"""
        height, width, _ = frame.shape
        
        # Draw full face mesh
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Draw eye contours more prominently
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Draw iris landmarks
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        
        # Draw EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ear > self.eye_closed_threshold else (0, 0, 255), 2)
    
    def _display_status(self, frame, is_drowsy, ear):
        """Display drowsiness status and metrics on the frame"""
        height, width, _ = frame.shape
        
        # Display eye status
        eye_status = "Eyes: Closed" if ear < self.eye_closed_threshold else "Eyes: Open"
        cv2.putText(frame, eye_status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if ear < self.eye_closed_threshold else (0, 255, 0), 2)
        
        # Display drowsiness status
        if is_drowsy:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add a red overlay to make the alert more visible
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Display closed duration if eyes are closed
        if self.closed_eye_start_time is not None:
            duration = time.time() - self.closed_eye_start_time
            cv2.putText(frame, f"Eyes Closed: {duration:.1f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if duration > 1.0 else (255, 0, 0), 2)
    
    def run(self):
        """Main method to start the drowsiness detection system"""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open video capture device")
            return
        
        # Get camera properties for optimization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera: {width}x{height} at {fps} FPS")
        
        frame_number = 0
        last_alarm_time = 0
        
        try:
            while True:
                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                frame_number += 1
                
                # Process the frame to detect drowsiness
                processed_frame, is_drowsy = self.process_frame(frame, frame_number)
                
                # Trigger alarm if drowsy (with cooldown to prevent continuous alarms)
                current_time = time.time()
                if is_drowsy and not self.alarm_active and (current_time - last_alarm_time) > 5:
                    last_alarm_time = current_time
                    # Start alarm in a separate thread
                    threading.Thread(target=self.play_alarm).start()
                
                # Display the processed frame
                cv2.imshow("Drowsiness Detection System", processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            logger.info("Drowsiness detection system shut down")

if __name__ == "__main__":
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        logging.error(f"Error starting drowsiness detector: {e}")
        import traceback
        traceback.print_exc()