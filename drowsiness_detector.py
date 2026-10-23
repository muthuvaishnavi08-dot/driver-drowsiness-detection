import cv2
import numpy as np
import time
import pygame
from scipy.spatial import distance as dist

# ---------------- CONFIGURATION ---------------- #
class Config:
    WIDTH = 640
    HEIGHT = 480
    EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold
    EYE_AR_CONSEC_FRAMES = 20 # Frames before alarm
    MOUTH_AR_THRESHOLD = 0.6
    ENHANCE_LIGHT = True

# ---------------- LOW LIGHT PROCESSING ---------------- #
class LowLightProcessor:
    @staticmethod
    def enhance_image(frame):
        """Enhances image for low light using CLAHE"""
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# ---------------- FACE DETECTION (OPENCV DNN) ---------------- #
class FaceDetector:
    def __init__(self):
        # Load Caffe model
        prototxt = "deploy.prototxt"
        caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
        
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def detect(self, frame):
        """Detect face using OpenCV DNN"""
        h, w = frame.shape[:2]
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Find largest face
        face_box = None
        max_conf = 0.5
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                face_box = (x, y, x2 - x, y2 - y)
                max_conf = confidence
                
        return face_box

# ---------------- DROWSINESS DETECTION ---------------- #
class DrowsinessDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        
        # States
        self.frame_counter = 0
        self.drowsiness_level = 0.0
        self.is_drowsy = False
        
        # Face landmark parameters (approximate for eye region)
        # Based on typical face proportions
        self.left_eye_ratio = (0.30, 0.35)  # x ratio, y ratio relative to face
        self.right_eye_ratio = (0.65, 0.35)
        
    def calculate_ear(self, eye_region):
        """Calculate EAR from eye region"""
        if eye_region is None or eye_region.shape[0] < 10:
            return 0.3
            
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate vertical extent (openness)
        top = np.min(gray[:, 1])
        bottom = np.max(gray[:, 1])
        height = bottom - top
        
        # Calculate horizontal extent
        left = np.min(gray[:, 0])
        right = np.max(gray[:, 0])
        width = right - left
        
        if width > 0:
            ear = height / width
            return ear
        return 0.3
        
    def extract_eyes(self, frame, face_box):
        """Extract eye regions from face"""
        if face_box is None:
            return None, None
            
        x, y, w, h = face_box
        
        # Eye regions (approximate positions)
        # Left eye: 25-40% horizontally, 20-40% vertically
        left_eye_x = int(x + w * 0.25)
        left_eye_y = int(y + h * 0.25)
        left_eye_w = int(w * 0.20)
        left_eye_h = int(h * 0.15)
        
        # Right eye
        right_eye_x = int(x + w * 0.55)
        right_eye_y = int(y + h * 0.25)
        right_eye_w = int(w * 0.20)
        right_eye_h = int(h * 0.15)
        
        left_eye = frame[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
        right_eye = frame[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
        
        return left_eye, right_eye

    def process(self, frame):
        """Main detection process"""
        original_frame = frame.copy()
        
        # 1. Low Light Enhancement
        if Config.ENHANCE_LIGHT:
            frame = LowLightProcessor.enhance_image(frame)

        # 2. Face Detection
        face_box = self.face_detector.detect(frame)
        
        if face_box is not None:
            x, y, w, h = face_box
            
            # Draw face box
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 3. Extract Eye Regions
            left_eye, right_eye = self.extract_eyes(original_frame, face_box)
            
            if left_eye is not None and right_eye is not None:
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                # 4. Drowsiness Logic
                if ear < Config.EAR_THRESHOLD:
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0
                    
                # Calculate drowsiness level
                if self.frame_counter > 5:
                    self.drowsiness_level = min(100, (self.frame_counter / Config.EYE_AR_CONSEC_FRAMES) * 100)
                    if self.drowsiness_level > 80:
                        self.is_drowsy = True
                else:
                    self.is_drowsy = False
                    self.drowsiness_level = max(0, self.drowsiness_level - 5)
                
                # 5. Visual Feedback
                color = (0, 255, 0)
                if self.is_drowsy:
                    color = (0, 0, 255)
                    cv2.putText(original_frame, "DROWSY! WAKE UP!", 
                              (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
                
                # Display metrics
                cv2.putText(original_frame, f"EAR: {ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(original_frame, f"Level: {self.drowsiness_level:.1f}%", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw eye boxes
                cv2.rectangle(original_frame, (x + int(w*0.25), y + int(h*0.25)), 
                            (x + int(w*0.45), y + int(h*0.40)), (255, 0, 0), 1)
                cv2.rectangle(original_frame, (x + int(w*0.55), y + int(h*0.25)), 
                            (x + int(w*0.75), y + int(h*0.40)), (255, 0, 0), 1)
        else:
            cv2.putText(original_frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return original_frame, self.is_drowsy, self.drowsiness_level