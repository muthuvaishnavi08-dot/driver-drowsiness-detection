import cv2
import numpy as np
import time
import platform
import winsound

# ==================== CONFIGURATION ==================== #
class Config:
    WIDTH = 640
    HEIGHT = 480
    
    # DROWSINESS
    EAR_THRESHOLD = 0.25
    EYE_AR_CONSEC_FRAMES = 12
    
    # YAWNING
    MAR_THRESHOLD = 0.40
    YOWN_CONSEC_FRAMES = 10
    
    ENHANCE_LIGHT = True

# ==================== LOW LIGHT ==================== #
def enhance_low_light(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# ==================== CALCULATE EAR ==================== #
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    if C > 0:
        return (A + B) / (2.0 * C)
    return 0.3

# ==================== AUDIO ALERT ==================== #
def play_alert_sound(is_drowsy):
    """Play beep sound based on alert type"""
    if is_drowsy:
        # Drowsy = high pitch urgent beep
        winsound.Beep(1500, 150)
    else:
        # Yawning = lower beep
        winsound.Beep(800, 200)

# ==================== DROWSINESS DETECTOR ==================== #
class DrowsinessDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        print("✓ Detectors loaded")
        
        self.eye_counter = 0
        self.yawn_counter = 0
        self.drowsiness_level = 0.0
        self.yawning_level = 0.0
        self.is_drowsy = False
        self.is_yawning = False
        self.ear = 0.3
        self.mar = 0.0
        
    def process(self, frame):
        original = frame.copy()
        
        if Config.ENHANCE_LIGHT:
            frame = enhance_low_light(frame)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            
            # ========== EYE DETECTION ==========
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                ear_values = []
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(original, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 255), 1)
                    
                    eye_points = np.array([
                        [x+ex, y+ey+eh//2],
                        [x+ex+ew//4, y+ey],
                        [x+ex+3*ew//4, y+ey],
                        [x+ex+ew, y+ey+eh//2],
                        [x+ex+ew//4, y+ey+eh],
                        [x+ex+3*ew//4, y+ey+eh]
                    ], dtype=np.int32)
                    
                    ear = calculate_ear(eye_points)
                    ear_values.append(ear)
                
                self.ear = np.mean(ear_values) if ear_values else 0.3
                
                if self.ear < Config.EAR_THRESHOLD:
                    self.eye_counter += 1
                else:
                    self.eye_counter = max(0, self.eye_counter - 1)
                
                self.drowsiness_level = min(100, (self.eye_counter / Config.EYE_AR_CONSEC_FRAMES) * 100)
                self.is_drowsy = self.drowsiness_level > 80
            else:
                self.eye_counter = max(0, self.eye_counter - 2)
                self.drowsiness_level = max(0, self.drowsiness_level - 5)
                self.is_drowsy = False
                self.ear = 0.3
            
            # ========== MOUTH DETECTION ==========
            mouth_y = int(y + h * 0.68)
            mouth_h = int(h * 0.32)
            mouth_x = int(x + w * 0.2)
            mouth_w = int(w * 0.6)
            
            mouth_roi = roi_gray[mouth_y-y:mouth_y-y+mouth_h, mouth_x-x:mouth_x-x+mouth_w]
            
            if mouth_roi.size > 0:
                _, mouth_binary = cv2.threshold(mouth_roi, 60, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mouth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x_m, y_m, w_m, h_m = cv2.boundingRect(largest)
                    
                    self.mar = h_m / max(w_m, 1)
                    
                    cv2.rectangle(original, (x+mouth_x+x_m, y+mouth_y+y_m), 
                              (x+mouth_x+x_m+w_m, y+mouth_y+y_m+h_m), (0, 0, 255), 1)
                    
                    if self.mar > Config.MAR_THRESHOLD:
                        self.yawn_counter += 1
                    else:
                        self.yawn_counter = max(0, self.yawn_counter - 1)
                    
                    self.yawning_level = min(100, (self.yawn_counter / Config.YOWN_CONSEC_FRAMES) * 100)
                    self.is_yawning = self.yawning_level > 80
                else:
                    self.yawn_counter = max(0, self.yawn_counter - 2)
                    self.yawning_level = max(0, self.yawning_level - 5)
                    self.is_yawning = False
                    self.mar = 0.0
            
            # ========== DISPLAY ==========
            if self.is_drowsy or self.is_yawning:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            if self.is_drowsy:
                status = "⚠️ DROWSY!"
            elif self.is_yawning:
                status = "⚠️ YAWNING!"
            else:
                status = "✅ OK"
            
            cv2.putText(original, status, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(original, f"EAR: {self.ear:.2f}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(original, f"MAR: {self.mar:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Level bars
            bar_y = 110
            cv2.rectangle(original, (10, bar_y), (210, bar_y + 18), (50, 50, 50), -1)
            cv2.rectangle(original, (10, bar_y), (10 + int(2 * self.drowsiness_level), bar_y + 18), color, -1)
            cv2.putText(original, f"Drowsy: {self.drowsiness_level:.0f}%", (220, bar_y + 14), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            bar_y2 = 135
            yawn_color = (0, 165, 255) if self.is_yawning else (50, 50, 50)
            cv2.rectangle(original, (10, bar_y2), (210, bar_y2 + 18), (50, 50, 50), -1)
            cv2.rectangle(original, (10, bar_y2), (10 + int(2 * self.yawning_level), bar_y2 + 18), yawn_color, -1)
            cv2.putText(original, f"Yawning: {self.yawning_level:.0f}%", (220, bar_y2 + 14), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        else:
            cv2.putText(original, "No face detected", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return original, (self.is_drowsy or self.is_yawning), self.drowsiness_level

# ==================== MAIN ==================== #
def main():
    print("=" * 50)
    print("  DRIVER DROWSINESS & YAWNING DETECTION")
    print("=" * 50)
    
    import winsound
    
    detector = DrowsinessDetector()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)
    
    # Track if alert was playing (to avoid continuous beeping)
    alert_playing = False
    
    print("\nStarting... Press Q to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        result_frame, is_alert, level = detector.process(frame)
        
        # ========== AUDIO ALERT ==========
        if detector.is_drowsy or detector.is_yawning:
            if not alert_playing:
                # Play alarm sound
                for _ in range(2):
                    winsound.Beep(1200, 1500)
                    time.sleep(0.1)
                    winsound.Beep(1000, 1500)
                alert_playing = True
        else:
            alert_playing = False
            
        cv2.imshow("Drowsiness Detection", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
