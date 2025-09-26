import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template_string, jsonify
import base64
import io

# --------------------------
# Flask Setup
# --------------------------
app = Flask(__name__)

# --------------------------
# Model + CSV Loading
# --------------------------
MODEL_PATH = "C:\\Users\\go1wt\OneDrive\\Desktop\\project\\Squadooz_hackathon\\Medical vision system\\kvasir_coarse_model.pth"
CSV_PATH = "remedy_mapping_dataset.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 8
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

remedy_df = pd.read_csv(CSV_PATH)

folder_to_condition = {
    "dyed-lifted-polyps": "polyp",
    "dyed-resection-margins": "polyp",
    "esophagitis": "esophagitis",
    "normal-cecum": "normal",
    "normal-pylorus": "normal",
    "normal-z-line": "normal",
    "polyps": "polyp",
    "ulcerative-colitis": "colitis"
}
class_names = list(folder_to_condition.keys())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# --------------------------
# HTML Template (Webcam + Capture)
# --------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🩺 Medical Vision System</title>
</head>
<body style="font-family:Arial; text-align:center; padding:20px;">
    <h2>🩺 Real-Time Medical Vision System</h2>

    <video id="video" width="400" height="300" autoplay></video>
    <br>
    <button onclick="capture()">📸 Capture & Analyze</button>

    <h3 id="result"></h3>

    <script>
        // Start webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                document.getElementById('video').srcObject = stream;
            })
            .catch(err => {
                alert("Camera access denied: " + err);
            });

        function capture() {
            let video = document.getElementById('video');
            let canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            let dataURL = canvas.toDataURL('image/png');

            fetch('/predict', {
                method: 'POST',
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('result').innerHTML = `
                    <b>Condition:</b> ${data.condition} <br>
                    <b>Confidence:</b> ${data.confidence} <br>
                    <b>Severity:</b> ${data.severity} <br>
                    <b>Urgency:</b> ${data.urgency} <br>
                    <b>Remedy:</b> ${data.remedy} <br>
                    <b>Specialist:</b> ${data.specialist}
                `;
            });
        }
    </script>
</body>
</html>
"""

# --------------------------
# Styled HTML Template
# --------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🩺 Medical Vision System</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            background: linear-gradient(135deg, #e0f7fa, #ffffff);
            color: #333;
            margin: 0;
            padding: 0;
        }
        header {
            background: #0288d1;
            color: white;
            padding: 20px;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .container {
            margin: 30px auto;
            max-width: 700px;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        video {
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            margin-bottom: 15px;
        }
        button {
            background: #0288d1;
            border: none;
            padding: 12px 25px;
            font-size: 16px;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background: #0277bd;
            transform: scale(1.05);
        }
        .result-card {
            margin-top: 20px;
            text-align: left;
            padding: 20px;
            border-radius: 12px;
            background: #f1faff;
            border-left: 6px solid #0288d1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .result-card h3 {
            margin-top: 0;
            color: #0288d1;
        }
        .result-card p {
            margin: 6px 0;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <header>🩺 Medical Vision System</header>
    <div class="container">
        <video id="video" width="500" height="350" autoplay></video>
        <br>
        <button onclick="capture()">📸 Capture & Analyze</button>

        <div id="result" class="result-card" style="display:none;">
            <h3>📌 Prediction Result</h3>
            <p><b>Condition:</b> <span id="cond"></span></p>
            <p><b>Confidence:</b> <span id="conf"></span></p>
            <p><b>Severity:</b> <span id="sev"></span></p>
            <p><b>Urgency:</b> <span id="urg"></span></p>
            <p><b>Remedy:</b> <span id="rem"></span></p>
            <p><b>Specialist:</b> <span id="spec"></span></p>
        </div>
    </div>

    <script>
        // Start webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                document.getElementById('video').srcObject = stream;
            })
            .catch(err => {
                alert("Camera access denied: " + err);
            });

        function capture() {
            let video = document.getElementById('video');
            let canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            let dataURL = canvas.toDataURL('image/png');

            fetch('/predict', {
                method: 'POST',
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('result').style.display = "block";
                document.getElementById('cond').innerText = data.condition;
                document.getElementById('conf').innerText = data.confidence;
                document.getElementById('sev').innerText = data.severity;
                document.getElementById('urg').innerText = data.urgency;
                document.getElementById('rem').innerText = data.remedy;
                document.getElementById('spec').innerText = data.specialist;
            });
        }
    </script>
</body>
</html>
"""


# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data"}), 400

    # Decode base64 image
    image_data = data["image"].split(",")[1]  # remove 'data:image/png;base64,'
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

    folder_label = class_names[predicted.item()]
    condition = folder_to_condition[folder_label]

    # Remedy lookup
    remedy_rows = remedy_df[remedy_df['condition_label'] == condition]
    if remedy_rows.empty:
        remedy_info = {
            'severity': 'Unknown',
            'urgency': 'Unknown',
            'remedy_text': 'No remedy found.',
            'recommended_specialist': 'N/A'
        }
    else:
        remedy_info = remedy_rows.sample(1).iloc[0]

    result = {
        "condition": condition,
        "confidence": f"{confidence:.2f}",
        "severity": remedy_info['severity'],
        "urgency": remedy_info['urgency'],
        "remedy": remedy_info['remedy_text'],
        "specialist": remedy_info['recommended_specialist']
    }

    return jsonify(result)

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
