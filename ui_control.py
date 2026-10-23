import cv2
import numpy as np

class UIController:
    def __init__(self):
        self.ear_threshold = 0.25
        
    def create_window(self):
        cv2.namedWindow("Settings")
        
    def update(self, frame):
        # Create a trackbar for EAR Threshold
        cv2.createTrackbar("EAR Threshold", "Settings", 25, 40, self.on_threshold_change)
        
        # Display on frame
        cv2.rectangle(frame, (frame.shape[1]-200, 0), (frame.shape[1], 130), (0,0,0), -1)
        cv2.putText(frame, f"EAR: {self.ear_threshold:.2f}", (frame.shape[1]-190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return frame
        
    def on_threshold_change(self, val):
        self.ear_threshold = val / 100.0