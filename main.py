# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import random
from PIL import Image

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Kvasir_folder = "kvasir"
Remedy_csv = "remedy_mapping_dataset.csv"

# Map Kvasir folders to coarse labels for remedy dataset
folder_to_condition = {
    "dyed-lifted-polyps": "polyp",
    "dyed-resection-margins": "polyp",
    "esophagitis": "inflammation",
    "normal-cecum": "normal",
    "normal-pylorus": "normal",
    "normal-z-line": "normal",
    "polyps": "polyp",
    "ulcerative-colitis": "inflammation"
}

# -----------------------------
# PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(Kvasir_folder, transform=transform)
class_names = dataset.classes  # folder names as labels
num_classes = len(class_names)
print(f"Classes found: {class_names}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# MODEL
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "kvasir_coarse_model.pth")
print("Training complete and model saved.")

# -----------------------------
# LOAD REMEDY DATASET
# -----------------------------
remedy_df = pd.read_csv(Remedy_csv)

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def predict_image(img_path):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        folder_label = class_names[pred_idx.item()]
        condition_label = folder_to_condition.get(folder_label, "normal")
    
    # Lookup remedies
    condition_rows = remedy_df[remedy_df['condition_label']==condition_label]
    if condition_rows.empty:
        remedy_info = {"severity":"unknown","urgency":"consult",
                       "remedy_text":"No data available","recommended_specialist":"N/A",
                       "confidence_threshold":0.7}
    else:
        # pick random matching severity for demo
        row = condition_rows.sample(n=1).iloc[0]
        remedy_info = row.to_dict()
    
    return {
        "folder_label": folder_label,
        "condition_label": condition_label,
        "confidence": float(conf.item()),
        **remedy_info
    }

# -----------------------------
# TEST INFERENCE
# -----------------------------
test_image_path = input("Enter path to test image: ")
result = predict_image(test_image_path)
print("\n===== Prediction Result =====")
for k,v in result.items():
    print(f"{k}: {v}")
