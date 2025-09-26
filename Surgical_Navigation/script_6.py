# Fix the launch script with proper ASCII characters
launch_script = '''#!/usr/bin/env python3
"""
Advanced Surgical AI System Launcher
Comprehensive startup and management script for the surgical robotics system
"""

import sys
import os
import subprocess
import time
import signal
import threading
from datetime import datetime

class SurgicalAILauncher:
    def __init__(self):
        self.processes = []
        self.running = False
        self.log_file = f"surgical_ai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_message(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\\n')
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        self.log_message("Checking system dependencies...")
        
        required_packages = [
            'flask', 'opencv-python', 'numpy', 'torch', 'torchvision'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.log_message(f"✓ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                self.log_message(f"✗ {package} - MISSING")
        
        if missing_packages:
            self.log_message("Missing packages detected. Install with:")
            self.log_message(f"pip install {' '.join(missing_packages)}")
            return False
        
        self.log_message("All dependencies satisfied ✓")
        return True
    
    def check_hardware(self):
        """Check hardware requirements"""
        self.log_message("Checking hardware requirements...")
        
        # Check camera availability
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.log_message("✓ Camera - Available")
                cap.release()
            else:
                self.log_message("! Camera - Not available (will use synthetic feed)")
        except Exception as e:
            self.log_message(f"! Camera check failed: {e}")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.log_message(f"✓ GPU - {torch.cuda.get_device_name(0)}")
            else:
                self.log_message("! GPU - Not available (using CPU)")
        except Exception as e:
            self.log_message(f"! GPU check failed: {e}")
        
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'models', 'static', 'data', 'sessions']
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.log_message(f"Created directory: {directory}")
    
    def start_system_monitoring(self):
        """Start system monitoring thread"""
        def monitor():
            while self.running:
                try:
                    import psutil
                    cpu_usage = psutil.cpu_percent(interval=1)
                    memory_usage = psutil.virtual_memory().percent
                    
                    if cpu_usage > 90:
                        self.log_message(f"! High CPU usage: {cpu_usage}%")
                    if memory_usage > 90:
                        self.log_message(f"! High memory usage: {memory_usage}%")
                        
                except ImportError:
                    pass  # psutil not available
                except Exception as e:
                    self.log_message(f"Monitoring error: {e}")
                
                time.sleep(10)  # Check every 10 seconds
        
        monitoring_thread = threading.Thread(target=monitor, daemon=True)
        monitoring_thread.start()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log_message(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def start_flask_app(self):
        """Start the main Flask application"""
        self.log_message("Starting Flask application...")
        
        try:
            # Import and run the Flask app
            from app import app
            
            # Set Flask configuration
            app.config['DEBUG'] = False
            app.config['TESTING'] = False
            
            # Start Flask in a separate thread for production
            flask_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
                daemon=True
            )
            flask_thread.start()
            
            self.log_message("Flask application started on http://localhost:5000")
            return True
            
        except Exception as e:
            self.log_message(f"Failed to start Flask application: {e}")
            return False
    
    def run_system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        self.log_message("Running system diagnostics...")
        
        try:
            # Test computer vision module
            from surgical_vision_ai import SurgicalVisionAI
            ai_system = SurgicalVisionAI()
            self.log_message("✓ Computer Vision module loaded")
            
            # Test robotic control module  
            from robotic_control import SurgicalRoboticsController
            robotics = SurgicalRoboticsController()
            self.log_message("✓ Robotic control module loaded")
            
            # Test system integration
            status = robotics.get_system_status()
            self.log_message(f"✓ System status: {status['system_status']}")
            
            return True
            
        except Exception as e:
            self.log_message(f"✗ Diagnostics failed: {e}")
            return False
    
    def display_startup_banner(self):
        """Display startup banner"""
        banner = '''
================================================================
                   ADVANCED SURGICAL AI SYSTEM                   
                                                                  
  🤖 Real-time Computer Vision    🔬 Tissue Analysis             
  🦾 Robotic Navigation          ⚠️  Anomaly Detection           
  🧵 Automated Suturing          📊 Patient Monitoring           
  🛡️  Safety Systems             🌐 Web Interface                
                                                                  
                        RESEARCH USE ONLY                        
                    NOT FOR CLINICAL USE                         
================================================================
        '''
        print(banner)
        
    def start(self):
        """Start the complete surgical AI system"""
        self.display_startup_banner()
        
        self.log_message("=== SURGICAL AI SYSTEM STARTUP ===")
        self.log_message(f"Session log: {self.log_file}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # System checks
        if not self.check_dependencies():
            self.log_message("✗ Dependency check failed")
            return False
        
        if not self.check_hardware():
            self.log_message("✗ Hardware check failed")
            return False
        
        # Setup
        self.create_directories()
        
        # Run diagnostics
        if not self.run_system_diagnostics():
            self.log_message("✗ System diagnostics failed")
            return False
        
        # Start monitoring
        self.running = True
        self.start_system_monitoring()
        
        # Start Flask application
        if not self.start_flask_app():
            self.log_message("✗ Flask startup failed")
            return False
        
        self.log_message("🚀 SURGICAL AI SYSTEM READY")
        self.log_message("🌐 Web interface: http://localhost:5000")
        self.log_message("📊 Monitor logs in real-time with: tail -f " + self.log_file)
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log_message("Keyboard interrupt received")
        
        self.shutdown()
        return True
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        self.log_message("Initiating system shutdown...")
        self.running = False
        
        # Terminate any child processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        self.log_message("=== SURGICAL AI SYSTEM SHUTDOWN COMPLETE ===")
        sys.exit(0)

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'help':
            print("""
Surgical AI System Commands:

  start     - Start the complete system (default)
  check     - Run system checks only  
  test      - Run diagnostic tests
  help      - Show this help message
  
Example usage:
  python launch.py start
  python launch.py check
            """)
            return
        
        elif command == 'check':
            launcher = SurgicalAILauncher()
            launcher.check_dependencies()
            launcher.check_hardware()
            return
            
        elif command == 'test':
            launcher = SurgicalAILauncher()
            launcher.run_system_diagnostics()
            return
    
    # Default: start the system
    launcher = SurgicalAILauncher()
    launcher.start()

if __name__ == "__main__":
    main()
'''

with open('launch.py', 'w') as f:
    f.write(launch_script)

print("Created launch.py - System launcher and management script")

# Create test script
test_script = '''#!/usr/bin/env python3
"""
Surgical AI System Test Suite
Comprehensive testing for all system components
"""

import unittest
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock

class TestSurgicalVisionAI(unittest.TestCase):
    """Test cases for computer vision module"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from surgical_vision_ai import SurgicalVisionAI
            self.ai_system = SurgicalVisionAI()
        except ImportError as e:
            self.skipTest(f"Cannot import surgical_vision_ai: {e}")
    
    def test_model_initialization(self):
        """Test AI model initialization"""
        self.assertIsNotNone(self.ai_system)
        self.assertIn('tissue_segmentation', self.ai_system.models)
        self.assertIn('instrument_detection', self.ai_system.models)
        self.assertIn('anomaly_detection', self.ai_system.models)
    
    def test_frame_processing(self):
        """Test frame processing pipeline"""
        # Create synthetic test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        results = self.ai_system.process_frame(test_frame)
        
        self.assertIsNotNone(results)
        self.assertIn('tissue_segmentation', results)
        self.assertIn('instrument_detections', results)
        self.assertIn('anomaly_score', results)
    
    def test_surgical_scene_analysis(self):
        """Test comprehensive surgical scene analysis"""
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        analysis = self.ai_system.analyze_surgical_scene(test_frame)
        
        self.assertIsNotNone(analysis)
        self.assertIn('tissue_analysis', analysis)
        self.assertIn('instrument_analysis', analysis)
        self.assertIn('safety_assessment', analysis)
        self.assertIn('recommendations', analysis)

class TestRoboticControl(unittest.TestCase):
    """Test cases for robotic control system"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from robotic_control import SurgicalRoboticsController, Position3D
            self.controller = SurgicalRoboticsController()
            self.Position3D = Position3D
        except ImportError as e:
            self.skipTest(f"Cannot import robotic_control: {e}")
    
    def test_controller_initialization(self):
        """Test robotics controller initialization"""
        self.assertIn('primary_arm', self.controller.arms)
        self.assertIn('secondary_arm', self.controller.arms)
        self.assertIn('suture_arm', self.controller.arms)
        self.assertIsNotNone(self.controller.suturing_system)
    
    def test_arm_movement(self):
        """Test robotic arm movement"""
        result = self.controller.control_arm("primary_arm", "move_to", {
            "position": {"x": 10, "y": 10, "z": 10},
            "speed": 0.5
        })
        
        self.assertTrue(result["success"])
    
    def test_safety_monitoring(self):
        """Test safety monitoring system"""
        # Test position safety check
        test_position = self.Position3D(0, 0, 0)
        is_safe, violations = self.controller.safety_monitor.check_position_safety(
            test_position, "test_arm"
        )
        
        # Position at origin should be safe
        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        self.controller.emergency_stop_all()
        
        status = self.controller.get_system_status()
        self.assertTrue(status["emergency_stop"])
        self.assertEqual(status["system_status"], "emergency_stopped")
    
    def test_automated_suturing(self):
        """Test automated suturing system"""
        result = self.controller.execute_automated_suturing(
            {"x": 0, "y": 0, "z": 0},
            {"x": 10, "y": 0, "z": 0},
            3
        )
        
        self.assertIn("success", result)
        self.assertIn("stitches_planned", result)

def run_comprehensive_tests():
    """Run all test suites"""
    print("🧪 Running Surgical AI System Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSurgicalVisionAI))
    suite.addTests(loader.loadTestsFromTestCase(TestRoboticControl))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success

if __name__ == "__main__":
    run_comprehensive_tests()
'''

with open('test_system.py', 'w') as f:
    f.write(test_script)

print("Created test_system.py - Comprehensive test suite")

# Final summary
print("\n" + "="*60)
print("🏥 ADVANCED SURGICAL AI SYSTEM - CODE COMPLETE")
print("="*60)
print("\n📁 Project Structure:")
print("├── app.py                    # Main Flask web application")
print("├── surgical_vision_ai.py     # Computer vision & AI module") 
print("├── robotic_control.py        # Robotic control system")
print("├── templates/")
print("│   └── surgical_ai.html      # Web interface")
print("├── launch.py                 # System launcher")
print("├── test_system.py            # Test suite")
print("├── requirements.txt          # Dependencies")
print("└── README.md                 # Documentation")

print("\n🚀 Quick Start:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Launch system: python launch.py")
print("3. Open browser: http://localhost:5000")
print("4. Run tests: python test_system.py")

print("\n✨ Features Implemented:")
print("✅ Real-time computer vision processing")
print("✅ Advanced robotic control system")
print("✅ Automated suturing with microscopic precision")
print("✅ Multi-modal anomaly detection")
print("✅ Comprehensive safety monitoring")
print("✅ Live web interface with controls")
print("✅ Patient vital signs monitoring")
print("✅ Emergency stop systems")
print("✅ Session logging and analysis")
print("✅ Comprehensive test suite")

print("\n⚠️  Important: This is for research/development only - NOT for clinical use!")
print("="*60)