import cv2
import numpy as np
import json
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
import random
from ultralytics import YOLO  # ✅ Real AI model for detection

app = Flask(__name__)

class SurgicalAISystem:
    def __init__(self):
        self.camera = None
        self.current_frame = None
        self.emergency_stop = False
        self.anomalies_detected = []

        # Patient vitals
        self.patient_vitals = {
            'heart_rate': 72,
            'spo2': 98,
            'blood_pressure_systolic': 120,
            'blood_pressure_diastolic': 80,
            'temperature': 98.6,
            'status': 'SAFE'
        }

        # Robotic arms (simulated)
        self.robotic_arms = {
            'arm_1': {'position': [0, 0, 0], 'status': 'ready', 'tool': 'scalpel'},
            'arm_2': {'position': [0, 0, 0], 'status': 'ready', 'tool': 'forceps'},
            'suture_arm': {'position': [0, 0, 0], 'status': 'ready', 'tool': 'needle_driver'}
        }

        # Surgery session info
        self.surgery_session = {
            'start_time': datetime.now(),
            'duration': 0,
            'status': 'active'
        }

        # ✅ Load YOLOv8 pretrained model
        self.model = YOLO("yolov8n.pt")  # you can replace with custom surgical model

        # Start monitoring threads
        self.start_monitoring_threads()

    def start_monitoring_threads(self):
        threading.Thread(target=self.monitor_vitals, daemon=True).start()
        threading.Thread(target=self.monitor_anomalies, daemon=True).start()

    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.camera = None
                return False
            return True
        except:
            self.camera = None
            return False

    def generate_synthetic_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def process_frame_with_ai(self, frame):
        """Use YOLOv8 for real detection"""
        results = self.model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def get_video_stream(self):
        while True:
            if self.emergency_stop:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "EMERGENCY STOP", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if not ret:
                        frame = self.generate_synthetic_frame()
                else:
                    frame = self.generate_synthetic_frame()

                frame = self.process_frame_with_ai(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    def monitor_vitals(self):
        while True:
            if not self.emergency_stop:
                self.patient_vitals['heart_rate'] = 72 + random.randint(-5, 5)
                self.patient_vitals['spo2'] = 98 + random.randint(-2, 2)
                self.patient_vitals['blood_pressure_systolic'] = 120 + random.randint(-10, 10)
                self.patient_vitals['blood_pressure_diastolic'] = 80 + random.randint(-5, 5)
                self.patient_vitals['temperature'] = 98.6 + random.uniform(-0.3, 0.3)

                if (self.patient_vitals['heart_rate'] > 100 or 
                    self.patient_vitals['spo2'] < 95 or 
                    self.patient_vitals['blood_pressure_systolic'] > 140):
                    self.patient_vitals['status'] = 'ALERT'
                else:
                    self.patient_vitals['status'] = 'SAFE'
            time.sleep(1)

    def monitor_anomalies(self):
        while True:
            if not self.emergency_stop and random.random() < 0.01:
                anomaly = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': random.choice(['Instrument Slip', 'Abnormal Movement']),
                    'severity': random.choice(['Low', 'Medium', 'High']),
                    'location': f"Region {random.randint(1, 4)}"
                }
                self.anomalies_detected.append(anomaly)
                if len(self.anomalies_detected) > 10:
                    self.anomalies_detected.pop(0)
            time.sleep(2)

    def control_robotic_arm(self, arm_id, action, parameters=None):
        if arm_id in self.robotic_arms and not self.emergency_stop:
            arm = self.robotic_arms[arm_id]
            if action == 'move' and parameters and 'position' in parameters:
                arm['position'] = parameters['position']
                arm['status'] = 'moving'
                threading.Timer(2.0, lambda: self.set_arm_status(arm_id, 'ready')).start()
            elif action == 'suture':
                arm['status'] = 'suturing'
                threading.Timer(5.0, lambda: self.set_arm_status(arm_id, 'ready')).start()
            elif action == 'stop':
                arm['status'] = 'stopped'
            return True
        return False

    def set_arm_status(self, arm_id, status):
        if arm_id in self.robotic_arms:
            self.robotic_arms[arm_id]['status'] = status

    def trigger_emergency_stop(self):
        self.emergency_stop = True
        for arm in self.robotic_arms.values():
            arm['status'] = 'emergency_stopped'
        self.patient_vitals['status'] = 'EMERGENCY'

    def reset_emergency_stop(self):
        self.emergency_stop = False
        for arm in self.robotic_arms.values():
            arm['status'] = 'ready'
        self.patient_vitals['status'] = 'SAFE'

# Initialize
surgical_ai = SurgicalAISystem()
surgical_ai.initialize_camera()

@app.route('/')
def index():
    return render_template('surgical_ai.html')

@app.route('/video_feed')
def video_feed():
    return Response(surgical_ai.get_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/vitals')
def get_vitals():
    return jsonify(surgical_ai.patient_vitals)

@app.route('/api/robotic_arms')
def get_robotic_arms():
    return jsonify(surgical_ai.robotic_arms)

@app.route('/api/anomalies')
def get_anomalies():
    return jsonify(surgical_ai.anomalies_detected)

@app.route('/api/control_arm', methods=['POST'])
def control_arm():
    data = request.json
    return jsonify({'success': surgical_ai.control_robotic_arm(data.get('arm_id'), data.get('action'), data.get('parameters'))})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    surgical_ai.trigger_emergency_stop()
    return jsonify({'status': 'emergency_stop_activated'})

@app.route('/api/reset_emergency', methods=['POST'])
def reset_emergency():
    surgical_ai.reset_emergency_stop()
    return jsonify({'status': 'emergency_reset'})

@app.route('/api/session_status')
def get_session_status():
    surgical_ai.surgery_session['duration'] = (datetime.now() - surgical_ai.surgery_session['start_time']).total_seconds()
    return jsonify(surgical_ai.surgery_session)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
