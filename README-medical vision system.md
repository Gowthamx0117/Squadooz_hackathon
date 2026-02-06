# 🩺 Medical Vision System

An AI-powered **Flask web app** that captures **real-time camera feed**, analyzes medical images, and provides prediction results such as **condition, confidence, severity, urgency, remedy, and specialist suggestion**.

---

## 🚀 Features
- 📸 **Live camera capture** directly from browser  
- 🧠 AI-ready backend (plug in your ML model)  
- 🎨 Styled frontend with **medical-themed UI**  
- ⚡ Real-time predictions (Flask + OpenCV + JavaScript)  

---

## 🛠️ Tech Stack
- **Backend**: Flask, OpenCV, NumPy  
- **Frontend**: HTML, CSS, JavaScript  
- **Model (optional)**: CNN / custom ML model  

---

## 📂 Project Structure
medical-vision-system/
├── app.py # Main Flask application
├── README.md # Project documentation
└── requirements.txt (optional - list of dependencies)


## 🔧 Installation & Setup
1. Clone this repository:
```bash
   git clone https://github.com/Gowthamx0117/medical-vision-system.git
   cd medical-vision-system## 🔧 Installation & Setup
```

### 1. Create a virtual environment (recommended)
  ```bash
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows
```

### 2. Install dependencies
  ```bash
  pip install flask opencv-python numpy
```

### 3. Run the Flask app:
  ```bash
  python flask-integrated.py
  Open browser and visit:
```

###   🖼️ Usage
   The app will open a live webcam feed in the browser.
   Click 📸 Capture & Analyze.

###   The app will process the captured frame and show:
```bash
   ✅ Condition
   📊 Confidence
   ⚠️ Severity
   🚑 Urgency
   💊 Remedy
   👨‍⚕️ Specialist Recommendation
```
##   📌 Future Improvements
```bash   
   🔬 Integrate a CNN-based medical image classifier
   🗂️ Store analysis results in a database
   🌐 Add user authentication and patient history
   📱 Deploy on cloud with mobile access
 ```  
   👨‍💻 Author
   Developed by Gowtham 🚀
