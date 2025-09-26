#!/usr/bin/env python3
"""
Advanced Surgical AI System Launcher
"""

import sys
import os
import time
import threading
from datetime import datetime

class SurgicalAILauncher:
    def __init__(self):
        self.running = False
        self.log_file = f"surgical_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log_message(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

    def check_dependencies(self):
        self.log_message("Checking dependencies...")
        required = ['flask', 'opencv-python', 'numpy', 'torch']
        missing = []

        for pkg in required:
            try:
                __import__(pkg.replace('-', '_'))
                self.log_message(f"✓ {pkg}")
            except ImportError:
                missing.append(pkg)
                self.log_message(f"✗ {pkg} - MISSING")

        if missing:
            self.log_message(f"Install missing: pip install {' '.join(missing)}")
            return False
        return True

    def start_flask_app(self):
        self.log_message("Starting Flask application...")
        try:
            from app import app
            flask_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=5000, debug=False),
                daemon=True
            )
            flask_thread.start()
            self.log_message("✓ Flask started on http://localhost:5000")
            return True
        except Exception as e:
            self.log_message(f"✗ Flask failed: {e}")
            return False

    def display_banner(self):
        print("""
================================================================
                ADVANCED SURGICAL AI SYSTEM
================================================================
  Real-time Computer Vision | Robotic Navigation
  Automated Suturing       | Anomaly Detection
  Patient Monitoring       | Safety Systems

                  RESEARCH USE ONLY
================================================================
        """)

    def start(self):
        self.display_banner()
        self.log_message("=== SYSTEM STARTUP ===")

        if not self.check_dependencies():
            return False

        if not self.start_flask_app():
            return False

        self.running = True
        self.log_message("🚀 SURGICAL AI SYSTEM READY")
        self.log_message("🌐 Web interface: http://localhost:5000")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log_message("Shutting down...")
            self.running = False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print("""
Surgical AI System Commands:
  start - Start the system (default)
  help  - Show this help
        """)
        return

    launcher = SurgicalAILauncher()
    launcher.start()

if __name__ == "__main__":
    main()
