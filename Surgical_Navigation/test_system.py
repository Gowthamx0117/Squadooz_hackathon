#!/usr/bin/env python3
"""
Surgical AI System Test Suite
"""

import unittest
import numpy as np

class TestSurgicalVisionAI(unittest.TestCase):
    def setUp(self):
        try:
            from surgical_vision_ai import SurgicalVisionAI
            self.ai_system = SurgicalVisionAI()
        except ImportError as e:
            self.skipTest(f"Cannot import: {e}")

    def test_initialization(self):
        self.assertIsNotNone(self.ai_system)
        self.assertIn('tissue_segmentation', self.ai_system.models)

    def test_frame_processing(self):
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.ai_system.process_frame(test_frame)
        self.assertIsNotNone(results)

class TestRoboticControl(unittest.TestCase):
    def setUp(self):
        try:
            from robotic_control import SurgicalRoboticsController
            self.controller = SurgicalRoboticsController()
        except ImportError as e:
            self.skipTest(f"Cannot import: {e}")

    def test_controller_init(self):
        self.assertIn('primary_arm', self.controller.arms)

    def test_arm_movement(self):
        result = self.controller.control_arm("primary_arm", "move_to", {
            "position": {"x": 10, "y": 10, "z": 10}
        })
        self.assertTrue(result["success"])

def main():
    print("🧪 Running Surgical AI Test Suite")
    print("=" * 40)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSurgicalVisionAI))
    suite.addTests(loader.loadTestsFromTestCase(TestRoboticControl))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nTests: {result.testsRun}, Failures: {len(result.failures)}")
    success = len(result.failures) == 0 and len(result.errors) == 0
    print("✅ PASSED" if success else "❌ FAILED")

if __name__ == "__main__":
    main()
