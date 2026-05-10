"""
====================================================================
ASTROGUARD V2.0: ADVANCED ASTRONAUT MONITORING SYSTEM
Real-Time Emotion + Physical Fatigue Detection
====================================================================
Features:
- Facial Expression Recognition
- Eye Blink Rate Analysis
- Yawn Detection
- Head Pose Tracking
- Posture Monitoring
- Fatigue Scoring
- Bone Density Risk Assessment
====================================================================
"""

import cv2
import numpy as np
from collections import deque
import time
import math

# ====================================================================
# ENHANCED ASTROGUARD WITH FATIGUE DETECTION
# ====================================================================

class AstroGuardAdvanced:
    """
    Advanced monitoring system for astronaut well-being
    Combines emotion detection with physical fatigue analysis
    """
    
    def __init__(self):
        print("="*70)
        print("🚀 ASTROGUARD V2.0 INITIALIZING...")
        print("="*70)
        
        # Emotions to detect
        self.emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Fearful', 'Disgusted']
        self.discomfort_emotions = ['Sad', 'Angry', 'Fearful', 'Disgusted']
        
        # Face detector
        print("[ASTROGUARD] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Eye detector
        print("[ASTROGUARD] Loading eye detector...")
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Smile detector
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Emotion history
        self.emotion_history = deque(maxlen=10)
        
        # FATIGUE MONITORING SYSTEMS
        print("[ASTROGUARD] Initializing fatigue detection...")
        
        # Blink detection
        self.blink_history = deque(maxlen=100)
        self.last_blink_time = time.time()
        self.blink_count = 0
        self.last_eye_state = "open"
        self.eye_aspect_ratio_threshold = 0.25
        
        # Yawn detection
        self.yawn_history = deque(maxlen=50)
        self.yawn_count = 0
        self.consecutive_yawn_frames = 0
        
        # Head pose tracking
        self.head_tilt_history = deque(maxlen=30)
        self.head_drop_count = 0
        
        # Posture tracking
        self.posture_history = deque(maxlen=50)
        self.poor_posture_count = 0
        
        # Fatigue scoring
        self.fatigue_score = 0
        self.fatigue_history = deque(maxlen=100)
        self.alert_level = "NORMAL"
        
        # Session statistics
        self.frame_count = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.start_time = time.time()
        
        # Bone density risk factors (microgravity simulation)
        self.session_duration_hours = 0
        self.physical_activity_score = 100  # Starts at 100%
        self.calcium_status = "NORMAL"
        
        print("[ASTROGUARD] ✅ All systems ready!")
        print("="*70)
    
    # ================================================================
    # EMOTION DETECTION (Original Functionality)
    # ================================================================
    
    def analyze_face_features(self, face_gray, face_color):
        """Analyze facial features to determine emotion"""
        h, w = face_gray.shape
        
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.3, 5)
        smiles = self.smile_cascade.detectMultiScale(face_gray, 1.8, 20)
        brightness = np.mean(face_gray)
        edges = cv2.Canny(face_gray, 50, 150)
        edge_density = np.sum(edges) / (h * w)
        
        probabilities = np.zeros(len(self.emotions))
        
        # Rule-based emotion detection
        if len(smiles) > 0:
            probabilities[1] = 0.7  # Happy
            probabilities[0] = 0.3
        elif len(eyes) == 2 and len(smiles) == 0:
            probabilities[0] = 0.6  # Neutral
            probabilities[1] = 0.2
            probabilities[2] = 0.2
        elif brightness < 100:
            probabilities[2] = 0.5  # Sad
            probabilities[0] = 0.3
            probabilities[5] = 0.2
        elif edge_density > 0.15:
            probabilities[3] = 0.6  # Angry
            probabilities[0] = 0.2
            probabilities[6] = 0.2
        elif len(eyes) >= 2 and brightness > 120:
            probabilities[4] = 0.5  # Surprised
            probabilities[1] = 0.3
            probabilities[0] = 0.2
        elif len(eyes) >= 2 and brightness < 110:
            probabilities[5] = 0.5  # Fearful
            probabilities[2] = 0.3
            probabilities[0] = 0.2
        else:
            probabilities[0] = 0.8
            probabilities[1] = 0.1
            probabilities[2] = 0.1
        
        probabilities = probabilities / np.sum(probabilities)
        noise = np.random.normal(0, 0.05, len(probabilities))
        probabilities = probabilities + noise
        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    # ================================================================
    # FATIGUE DETECTION SYSTEMS
    # ================================================================
    
    def calculate_eye_aspect_ratio(self, eye_region):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        h, w = eye_region.shape
        
        # Simplified EAR calculation based on eye region dimensions
        # Lower values indicate closed/closing eyes
        brightness = np.mean(eye_region)
        vertical_ratio = h / max(w, 1)
        
        # Estimate EAR
        ear = (brightness / 255.0) * vertical_ratio
        return ear
    
    def detect_blink(self, eyes, face_gray):
        """Detect eye blinks"""
        current_time = time.time()
        
        if len(eyes) == 0:
            return False
        
        # Analyze each detected eye
        ear_values = []
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_gray[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                ear = self.calculate_eye_aspect_ratio(eye_region)
                ear_values.append(ear)
        
        if not ear_values:
            return False
        
        avg_ear = np.mean(ear_values)
        
        # Detect blink (eyes closed)
        is_blink = avg_ear < self.eye_aspect_ratio_threshold
        
        # Blink state change detection
        if is_blink and self.last_eye_state == "open":
            self.blink_count += 1
            self.last_blink_time = current_time
            self.last_eye_state = "closed"
            return True
        elif not is_blink:
            self.last_eye_state = "open"
        
        return False
    
    def calculate_blink_rate(self):
        """Calculate blinks per minute"""
        current_time = time.time()
        elapsed_minutes = (current_time - self.start_time) / 60.0
        
        if elapsed_minutes < 0.1:  # Prevent division by zero
            return 0
        
        blink_rate = self.blink_count / elapsed_minutes
        
        # Normal blink rate: 15-20 per minute
        # Fatigue indicators:
        # - Very low (<10): Concentration or microsleep
        # - Very high (>30): Eye strain or fatigue
        
        return blink_rate
    
    def detect_yawn(self, face_gray, mouth_region=None):
        """Detect yawning (mouth wide open)"""
        h, w = face_gray.shape
        
        # Focus on lower half of face for mouth detection
        lower_face = face_gray[int(h*0.5):, :]
        
        # Detect large dark regions (open mouth)
        _, thresh = cv2.threshold(lower_face, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for large mouth opening
        is_yawning = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (h * w * 0.05):  # Mouth area > 5% of face
                x, y, mw, mh = cv2.boundingRect(contour)
                aspect_ratio = mh / max(mw, 1)
                
                # Yawn detection: wide vertical opening
                if aspect_ratio > 0.8 and area > (h * w * 0.08):
                    is_yawning = True
                    self.consecutive_yawn_frames += 1
                    break
        
        if not is_yawning:
            self.consecutive_yawn_frames = 0
        
        # Confirm yawn after multiple consecutive frames
        if self.consecutive_yawn_frames > 5:
            if len(self.yawn_history) == 0 or self.yawn_history[-1] < self.frame_count - 30:
                self.yawn_count += 1
                self.yawn_history.append(self.frame_count)
                return True
        
        return False
    
    def analyze_head_pose(self, face_rect, frame_shape):
        """Analyze head position and tilt"""
        x, y, w, h = face_rect
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate face center
        face_center_y = y + h // 2
        frame_center_y = frame_h // 2
        
        # Vertical position (head dropping)
        vertical_deviation = (face_center_y - frame_center_y) / frame_h
        
        # Head tilt detection (using face rectangle aspect ratio)
        aspect_ratio = h / max(w, 1)
        
        # Calculate head drop score
        head_drop_score = 0
        
        # Head too low (nodding off)
        if vertical_deviation > 0.15:
            head_drop_score += 2
            self.head_drop_count += 1
        
        # Unusual tilt
        if aspect_ratio < 1.2 or aspect_ratio > 1.8:
            head_drop_score += 1
        
        self.head_tilt_history.append(head_drop_score)
        
        return head_drop_score
    
    def analyze_posture(self, face_rect, frame_shape):
        """Analyze body posture based on face position"""
        x, y, w, h = face_rect
        frame_h, frame_w = frame_shape[:2]
        
        # Horizontal position (slouching/leaning)
        face_center_x = x + w // 2
        frame_center_x = frame_w // 2
        horizontal_deviation = abs(face_center_x - frame_center_x) / frame_w
        
        # Face size (distance from camera - leaning forward/back)
        face_area = w * h
        frame_area = frame_w * frame_h
        relative_size = face_area / frame_area
        
        # Posture scoring
        posture_score = 0
        
        # Too far from center (slouching/leaning)
        if horizontal_deviation > 0.2:
            posture_score += 2
            self.poor_posture_count += 1
        
        # Too close or too far (poor distance)
        if relative_size < 0.05 or relative_size > 0.25:
            posture_score += 1
        
        self.posture_history.append(posture_score)
        
        return posture_score
    
    # ================================================================
    # FATIGUE SCORING AND ANALYSIS
    # ================================================================
    
    def calculate_fatigue_score(self, blink_rate, yawn_detected, head_drop, posture):
        """Calculate comprehensive fatigue score (0-100)"""
        score = 0
        
        # Blink rate analysis (30 points max)
        if blink_rate < 10:
            score += 20  # Very low - possible microsleep
        elif blink_rate > 30:
            score += 15  # Very high - eye strain
        elif blink_rate < 12 or blink_rate > 25:
            score += 10  # Abnormal but not critical
        
        # Yawn detection (25 points max)
        recent_yawns = sum(1 for y in self.yawn_history if self.frame_count - y < 1800)  # Last 60 sec
        score += min(recent_yawns * 8, 25)
        
        # Head position (25 points max)
        if len(self.head_tilt_history) > 0:
            avg_head_drop = np.mean(list(self.head_tilt_history)[-30:])
            score += min(avg_head_drop * 8, 25)
        
        # Posture analysis (20 points max)
        if len(self.posture_history) > 0:
            avg_posture = np.mean(list(self.posture_history)[-30:])
            score += min(avg_posture * 7, 20)
        
        # Smooth the score
        self.fatigue_history.append(score)
        if len(self.fatigue_history) > 0:
            score = np.mean(list(self.fatigue_history)[-20:])
        
        return min(score, 100)
    
    def get_alert_level(self, fatigue_score):
        """Determine alert level based on fatigue score"""
        if fatigue_score < 25:
            return "NORMAL", (0, 255, 0)
        elif fatigue_score < 50:
            return "CAUTION", (0, 255, 255)
        elif fatigue_score < 75:
            return "WARNING", (0, 165, 255)
        else:
            return "CRITICAL", (0, 0, 255)
    
    def assess_bone_density_risk(self):
        """Assess bone density risk based on session duration and activity"""
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        
        # Simulate activity degradation
        self.physical_activity_score = max(100 - (elapsed_hours * 2), 50)
        
        # Risk assessment
        if elapsed_hours > 2 and self.physical_activity_score < 70:
            risk = "ELEVATED"
            self.calcium_status = "MONITOR"
        elif elapsed_hours > 4:
            risk = "HIGH"
            self.calcium_status = "INTERVENTION_NEEDED"
        else:
            risk = "LOW"
            self.calcium_status = "NORMAL"
        
        return risk, self.calcium_status
    
    # ================================================================
    # VISUALIZATION AND REPORTING
    # ================================================================
    
    def draw_fatigue_panel(self, frame, fatigue_score, blink_rate, yawn_count):
        """Draw comprehensive fatigue monitoring panel"""
        panel_x = frame.shape[1] - 400
        panel_y = 100
        panel_w = 380
        panel_h = 500
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "FATIGUE ANALYSIS", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = panel_y + 60
        
        # Fatigue Score
        alert_level, color = self.get_alert_level(fatigue_score)
        cv2.putText(frame, f"Fatigue Score: {fatigue_score:.1f}/100", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Fatigue bar
        bar_w = int((fatigue_score / 100) * (panel_w - 40))
        cv2.rectangle(frame, (panel_x + 10, y_offset + 10),
                     (panel_x + panel_w - 10, y_offset + 30),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (panel_x + 10, y_offset + 10),
                     (panel_x + 10 + bar_w, y_offset + 30),
                     color, -1)
        
        y_offset += 50
        
        # Alert Level
        cv2.putText(frame, f"Alert Level: {alert_level}", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        y_offset += 40
        
        # Metrics
        metrics = [
            f"Blink Rate: {blink_rate:.1f}/min",
            f"  Normal: 15-20/min",
            f"Yawns: {yawn_count}",
            f"Head Drops: {self.head_drop_count}",
            f"Posture Issues: {self.poor_posture_count}"
        ]
        
        for metric in metrics:
            cv2.putText(frame, metric, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        y_offset += 20
        
        # Bone density assessment
        bone_risk, calcium = self.assess_bone_density_risk()
        cv2.putText(frame, "BONE DENSITY RISK:", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        risk_color = (0, 255, 0) if bone_risk == "LOW" else (0, 165, 255) if bone_risk == "ELEVATED" else (0, 0, 255)
        cv2.putText(frame, f"Risk Level: {bone_risk}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)
        y_offset += 25
        
        cv2.putText(frame, f"Activity Score: {self.physical_activity_score:.0f}%", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(frame, f"Calcium: {calcium}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recommendations
        if fatigue_score > 50:
            y_offset += 40
            cv2.putText(frame, "⚠️ RECOMMENDATIONS:", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            y_offset += 25
            
            recs = []
            if blink_rate < 10:
                recs.append("Take immediate break")
            if yawn_count > 3:
                recs.append("Rest required")
            if fatigue_score > 75:
                recs.append("STOP OPERATIONS")
            
            for rec in recs[:3]:
                cv2.putText(frame, f"• {rec}", (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
    
    def draw_emotion_bars(self, frame, probabilities):
        """Draw emotion probability bars"""
        x_offset = 10
        y_offset = 100
        bar_height = 20
        bar_spacing = 30
        max_bar_width = 300
        
        for i, (emotion, prob) in enumerate(zip(self.emotions, probabilities)):
            y = y_offset + i * bar_spacing
            
            cv2.rectangle(frame, (x_offset, y), 
                         (x_offset + max_bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            bar_width = int(prob * max_bar_width)
            color = (0, 255, 0) if emotion not in self.discomfort_emotions else (0, 0, 255)
            cv2.rectangle(frame, (x_offset, y), 
                         (x_offset + bar_width, y + bar_height),
                         color, -1)
            
            text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x_offset + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # ================================================================
    # MAIN DETECTION LOOP
    # ================================================================
    
    def run_detection(self):
        """Main detection loop with emotion and fatigue monitoring"""
        print("\n[ASTROGUARD] Starting camera...")
        print("[ASTROGUARD] Monitoring:")
        print("  ✓ Facial Expressions")
        print("  ✓ Eye Blink Rate")
        print("  ✓ Yawn Detection")
        print("  ✓ Head Position")
        print("  ✓ Posture Analysis")
        print("  ✓ Bone Density Risk")
        print("\n[ASTROGUARD] Controls:")
        print("  Q: Quit | S: Screenshot | R: Reset")
        print("="*70)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Cannot access camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("[ASTROGUARD] ✅ All systems operational!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                
                # Emotion analysis
                probabilities = self.analyze_face_features(face_gray, face_color)
                emotion_idx = np.argmax(probabilities)
                emotion = self.emotions[emotion_idx]
                confidence = probabilities[emotion_idx]
                
                # Fatigue detection
                eyes = self.eye_cascade.detectMultiScale(face_gray, 1.3, 5)
                self.detect_blink(eyes, face_gray)
                yawn_detected = self.detect_yawn(face_gray)
                head_drop = self.analyze_head_pose((x, y, w, h), frame.shape)
                posture = self.analyze_posture((x, y, w, h), frame.shape)
                
                # Calculate metrics
                blink_rate = self.calculate_blink_rate()
                self.fatigue_score = self.calculate_fatigue_score(
                    blink_rate, yawn_detected, head_drop, posture
                )
                
                # Draw face rectangle
                is_discomfort = emotion in self.discomfort_emotions
                alert_level, alert_color = self.get_alert_level(self.fatigue_score)
                
                # Color based on worse condition
                if self.fatigue_score > 50 or is_discomfort:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Labels
                cv2.putText(frame, f"{emotion}: {confidence*100:.0f}%", 
                           (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Fatigue: {self.fatigue_score:.0f}%", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
                
                # Critical alerts
                if self.fatigue_score > 75:
                    cv2.putText(frame, "⚠️ CRITICAL FATIGUE - STOP OPERATIONS", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Draw panels
                self.draw_emotion_bars(frame, probabilities)
                self.draw_fatigue_panel(frame, self.fatigue_score, blink_rate, self.yawn_count)
            
            # Header
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            cv2.putText(frame, "ASTROGUARD V2.0", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "Emotion + Fatigue + Bone Density Monitor", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('ASTROGUARD V2.0 - Advanced Astronaut Monitoring', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'astroguard_{int(time.time())}.jpg', frame)
                print("[ASTROGUARD] Screenshot saved")
            elif key == ord('r'):
                self.__init__()
                print("[ASTROGUARD] System reset")
        
        cap.release()
        cv2.destroyAllWindows()
        self.display_summary()
    
    def display_summary(self):
        """Display comprehensive session summary"""
        print("\n" + "="*70)
        print("📊 ASTROGUARD V2.0 SESSION SUMMARY")
        print("="*70)
        
        elapsed = int(time.time() - self.start_time)
        print(f"\nSession Duration: {elapsed//60:02d}:{elapsed%60:02d}")
        print(f"Total Frames: {self.frame_count}")
        
        print("\n--- FATIGUE METRICS ---")
        print(f"Final Fatigue Score: {self.fatigue_score:.1f}/100")
        print(f"Blink Rate: {self.calculate_blink_rate():.1f}/min (Normal: 15-20)")
        print(f"Total Yawns: {self.yawn_count}")
        print(f"Head Drops: {self.head_drop_count}")
        print(f"Posture Issues: {self.poor_posture_count}")
        
        print("\n--- BONE DENSITY ASSESSMENT ---")
        risk, calcium = self.assess_bone_density_risk()
        print(f"Risk Level: {risk}")
        print(f"Activity Score: {self.physical_activity_score:.0f}%")
        print(f"Calcium Status: {calcium}")
        
        if self.fatigue_score > 50 or risk != "LOW":
            print("\n⚠️ RECOMMENDATIONS:")
            if self.fatigue_score > 50:
                print("  • Schedule mandatory rest period")
            if risk != "LOW":
                print("  • Increase resistance exercise")
                print("  • Review calcium supplementation")
            if self.yawn_count > 5:
                print("  • Evaluate sleep quality")
        
        print("\n" + "="*70)
        print("🚀 SESSION COMPLETE")
        print("="*70)

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    🚀 ASTROGUARD V2.0 🚀                           ║")
    print("║                                                                    ║")
    print("║        Advanced Astronaut Monitoring System                       ║")
    print("║     Emotion • Fatigue • Bone Density Detection                    ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════