import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import sys

# --------------------------
# Load trained model
# --------------------------
model_path = "kvasir_coarse_model.pth"   # saved model file
csv_path = "remedy_mapping_dataset.csv"

if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' not found.")
    sys.exit(1)
if not os.path.exists(csv_path):
    print(f"❌ Error: CSV file '{csv_path}' not found.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 8
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

model = model.to(device)
model.eval()

# --------------------------
# Remedy Mapping Dataset
# --------------------------
try:
    remedy_df = pd.read_csv(csv_path)
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

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

# --------------------------
# Image Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# --------------------------
# Webcam Capture on Demand
# --------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    sys.exit(1)

print("✅ Webcam started. Press SPACE to capture and analyze, 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Show live preview only
    cv2.imshow("🩺 Medical Vision System - Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == 32:  # SPACE pressed → capture & analyze
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

            folder_label = class_names[predicted.item()]
            condition = folder_to_condition[folder_label]

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

            # Display result on the same frame
            text = f"{condition} | Sev:{remedy_info['severity']} | Conf:{confidence:.2f}"
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("🩺 Medical Vision System - Analysis", frame)

            # Print detailed result in terminal
            print(f"""
            📌 Prediction
            -------------
            Folder Label : {folder_label}
            Condition    : {condition}
            Confidence   : {confidence:.2f}

            🩺 Remedy Suggestion
            ---------------------
            Severity     : {remedy_info['severity']}
            Urgency      : {remedy_info['urgency']}
            Remedy       : {remedy_info['remedy_text']} 
            Specialist   : {remedy_info['recommended_specialist']}
            """)

        except Exception as e:
            print(f"❌ Error during inference: {e}")

cap.release()
cv2.destroyAllWindows()
