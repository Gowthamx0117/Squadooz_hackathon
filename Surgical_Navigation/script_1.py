# Create the HTML template for the surgical AI interface
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Surgical AI System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #ffffff;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .header h1 {
            font-size: 2.5em;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .header p {
            color: #cccccc;
            margin-top: 5px;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 120px);
        }
        
        .video-section {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        .video-container {
            position: relative;
            width: 100%;
            height: 70%;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        
        #videoFeed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .video-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .controls-section {
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .control-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .control-btn.primary {
            background: #00ff88;
            color: #000;
        }
        
        .control-btn.primary:hover {
            background: #00cc66;
            transform: translateY(-2px);
        }
        
        .control-btn.danger {
            background: #ff4444;
            color: #fff;
        }
        
        .control-btn.danger:hover {
            background: #cc3333;
            transform: translateY(-2px);
        }
        
        .control-btn.warning {
            background: #ffaa00;
            color: #000;
        }
        
        .control-btn.warning:hover {
            background: #cc8800;
            transform: translateY(-2px);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #00ff88;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 5px;
        }
        
        .vitals-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .vital-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .vital-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ff88;
        }
        
        .vital-label {
            font-size: 0.8em;
            color: #cccccc;
            margin-top: 5px;
        }
        
        .status-indicator {
            width: 100%;
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .status-safe {
            background: #00ff88;
            color: #000;
        }
        
        .status-alert {
            background: #ff4444;
            color: #fff;
            animation: pulse 1s infinite;
        }
        
        .status-emergency {
            background: #ff0000;
            color: #fff;
            animation: pulse 0.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .robotic-arm {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .arm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .arm-status {
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .status-ready { background: #00ff88; color: #000; }
        .status-moving { background: #ffaa00; color: #000; }
        .status-suturing { background: #0088ff; color: #fff; }
        .status-stopped { background: #ff4444; color: #fff; }
        .status-emergency_stopped { background: #ff0000; color: #fff; }
        
        .anomaly-item {
            background: rgba(255, 68, 68, 0.2);
            border-left: 4px solid #ff4444;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        
        .anomaly-header {
            display: flex;
            justify-content: space-between;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .severity-high { color: #ff4444; }
        .severity-medium { color: #ffaa00; }
        .severity-low { color: #00ff88; }
        
        .session-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .emergency-panel {
            background: rgba(255, 0, 0, 0.2);
            border: 2px solid #ff4444;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        
        .emergency-btn {
            width: 100%;
            padding: 20px;
            font-size: 1.5em;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            background: #ff4444;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .emergency-btn:hover {
            background: #cc3333;
            transform: scale(1.05);
        }
        
        .reset-btn {
            margin-top: 10px;
            width: 100%;
            padding: 10px;
            background: #00ff88;
            color: #000;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .scrollable {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .scrollable::-webkit-scrollbar {
            width: 6px;
        }
        
        .scrollable::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        .scrollable::-webkit-scrollbar-thumb {
            background: #00ff88;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Advanced Surgical AI System</h1>
        <p>Real-time Computer Vision • Robotic Navigation • Tissue Analysis • Anomaly Detection</p>
    </div>
    
    <div class="main-container">
        <div class="video-section">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Live Surgical Feed">
                <div class="video-overlay">
                    <div>🎥 Live Feed Active</div>
                    <div>📊 AI Processing: ON</div>
                    <div>🔍 Segmentation: Active</div>
                </div>
            </div>
            
            <div class="controls-section">
                <button class="control-btn primary" onclick="controlArm('arm_1', 'move')">Move Arm 1</button>
                <button class="control-btn primary" onclick="controlArm('arm_2', 'move')">Move Arm 2</button>
                <button class="control-btn warning" onclick="controlArm('suture_arm', 'suture')">Auto Suture</button>
                <button class="control-btn primary" onclick="location.reload()">Refresh Feed</button>
            </div>
        </div>
        
        <div class="sidebar">
            <!-- Patient Vitals Panel -->
            <div class="panel">
                <h3>👤 Patient Vitals</h3>
                <div id="statusIndicator" class="status-indicator status-safe">SAFE</div>
                <div class="vitals-grid">
                    <div class="vital-item">
                        <div id="heartRate" class="vital-value">72</div>
                        <div class="vital-label">Heart Rate (bpm)</div>
                    </div>
                    <div class="vital-item">
                        <div id="spo2" class="vital-value">98</div>
                        <div class="vital-label">SpO₂ (%)</div>
                    </div>
                    <div class="vital-item">
                        <div id="bpSystolic" class="vital-value">120</div>
                        <div class="vital-label">BP Systolic</div>
                    </div>
                    <div class="vital-item">
                        <div id="bpDiastolic" class="vital-value">80</div>
                        <div class="vital-label">BP Diastolic</div>
                    </div>
                </div>
            </div>
            
            <!-- Robotic Arms Panel -->
            <div class="panel">
                <h3>🦾 Robotic Arms</h3>
                <div id="roboticArms">
                    <!-- Robotic arms will be populated by JavaScript -->
                </div>
            </div>
            
            <!-- Anomaly Detection Panel -->
            <div class="panel">
                <h3>⚠️ Anomaly Detection</h3>
                <div id="anomalies" class="scrollable">
                    <div style="text-align: center; color: #00ff88;">No anomalies detected</div>
                </div>
            </div>
            
            <!-- Emergency Controls -->
            <div class="emergency-panel">
                <h3 style="color: #ff4444; margin-bottom: 15px;">🚨 Emergency Controls</h3>
                <button class="emergency-btn" onclick="emergencyStop()">EMERGENCY STOP</button>
                <button class="reset-btn" onclick="resetEmergency()">Reset Emergency</button>
            </div>
            
            <!-- Session Info -->
            <div class="panel">
                <h3>📊 Session Info</h3>
                <div class="session-info">
                    <div class="vital-item">
                        <div id="sessionDuration" class="vital-value">0:00</div>
                        <div class="vital-label">Duration</div>
                    </div>
                    <div class="vital-item">
                        <div id="sessionStatus" class="vital-value">Active</div>
                        <div class="vital-label">Status</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update data every second
        function updateData() {
            // Update vitals
            fetch('/api/vitals')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('heartRate').textContent = data.heart_rate;
                    document.getElementById('spo2').textContent = data.spo2;
                    document.getElementById('bpSystolic').textContent = data.blood_pressure_systolic;
                    document.getElementById('bpDiastolic').textContent = data.blood_pressure_diastolic;
                    
                    const statusElement = document.getElementById('statusIndicator');
                    statusElement.textContent = data.status;
                    statusElement.className = 'status-indicator status-' + data.status.toLowerCase();
                });
            
            // Update robotic arms
            fetch('/api/robotic_arms')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('roboticArms');
                    container.innerHTML = '';
                    Object.entries(data).forEach(([armId, armData]) => {
                        const armDiv = document.createElement('div');
                        armDiv.className = 'robotic-arm';
                        armDiv.innerHTML = `
                            <div class="arm-header">
                                <strong>${armId.replace('_', ' ').toUpperCase()}</strong>
                                <span class="arm-status status-${armData.status}">${armData.status.toUpperCase()}</span>
                            </div>
                            <div style="font-size: 0.9em; color: #cccccc;">
                                Tool: ${armData.tool} | Position: [${armData.position.join(', ')}]
                            </div>
                        `;
                        container.appendChild(armDiv);
                    });
                });
            
            // Update anomalies
            fetch('/api/anomalies')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('anomalies');
                    if (data.length === 0) {
                        container.innerHTML = '<div style="text-align: center; color: #00ff88;">No anomalies detected</div>';
                    } else {
                        container.innerHTML = '';
                        data.slice(-5).forEach(anomaly => {
                            const anomalyDiv = document.createElement('div');
                            anomalyDiv.className = 'anomaly-item';
                            anomalyDiv.innerHTML = `
                                <div class="anomaly-header">
                                    <span>${anomaly.type}</span>
                                    <span class="severity-${anomaly.severity.toLowerCase()}">${anomaly.severity}</span>
                                </div>
                                <div style="font-size: 0.8em;">
                                    ${anomaly.location} • ${anomaly.timestamp}
                                </div>
                            `;
                            container.appendChild(anomalyDiv);
                        });
                    }
                });
            
            // Update session info
            fetch('/api/session_status')
                .then(response => response.json())
                .then(data => {
                    const minutes = Math.floor(data.duration / 60);
                    const seconds = Math.floor(data.duration % 60);
                    document.getElementById('sessionDuration').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    document.getElementById('sessionStatus').textContent = data.status;
                });
        }
        
        function controlArm(armId, action) {
            const parameters = action === 'move' ? {position: [Math.random() * 10, Math.random() * 10, Math.random() * 10]} : null;
            
            fetch('/api/control_arm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    arm_id: armId,
                    action: action,
                    parameters: parameters
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`${action} command sent to ${armId}`);
                }
            });
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to activate EMERGENCY STOP?')) {
                fetch('/api/emergency_stop', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    alert('EMERGENCY STOP ACTIVATED!');
                });
            }
        }
        
        function resetEmergency() {
            if (confirm('Are you sure you want to reset the emergency stop?')) {
                fetch('/api/reset_emergency', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    alert('Emergency reset successful. System resumed.');
                });
            }
        }
        
        // Start updating data
        updateData();
        setInterval(updateData, 1000);
        
        // Add some visual feedback for interactions
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 150);
            });
        });
    </script>
</body>
</html>
'''

# Create templates directory and save the HTML file
import os
os.makedirs('templates', exist_ok=True)

with open('templates/surgical_ai.html', 'w') as f:
    f.write(html_template)

print("Created templates/surgical_ai.html - Advanced surgical AI web interface")