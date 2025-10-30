<h1 align="center">🧠 NeuroScan AI — Brain Tumor Detection using EfficientNet-B0</h1> <p align="center"> NeuroScan AI is an advanced medical imaging web application built with <b>Streamlit</b>, designed to assist clinicians and researchers in brain tumor detection using MRI scans.<br> It classifies MRI images into four categories — <b>Glioma</b>, <b>Meningioma</b>, <b>Pituitary</b>, and <b>No Tumor</b> — and provides <b>Grad-CAM</b> visual explanations for model interpretability. </p>

🚀 Features
🧩 EfficientNet-B0–based deep learning model for tumor classification
🔥 Grad-CAM visualization for interpretable AI predictions
💻 Streamlit-powered interactive interface
🌐 Modern Medicio-style UI (teal/blue theme, responsive, minimal)
📊 Displays prediction confidence for each tumor class

🧾 Clean layout with sections for Features, About, Predictor, Samples, and FAQ

🗂️ Project Structure
BRAIN_TUMOR/
│
├── Axial.png
├── Brainscan.png
├── Radiology.png
├── bg_network.png
├── hero_brain_tube.png
├── workflow.png
├── logo.png
├── efficientnet_b0_best.pth        ← trained model weights
│
├── neuroscan_app.py                 ← main Streamlit app file
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation & Setup
🧭 Step 1 — Clone the Repository
git clone https://github.com/YOUR-USERNAME/NeuroScan-AI.git
cd NeuroScan-AI

🧭 Step 2 — Create a Virtual Environment (⚠️ Required)
Creating a virtual environment ensures package isolation.
For Windows
python -m venv venv
venv\Scripts\activate

For macOS / Linux
python3 -m venv venv
source venv/bin/activate

⚠️ Note: You must activate the virtual environment before installing dependencies, otherwise the app may fail due to version conflicts.
🧭 Step 3 — Install Dependencies
pip install -r requirements.txt

🧭 Step 4 — Run the Streamlit App
streamlit run neuroscan_app.py

The app will start at:
👉 http://localhost:8501

🧩 Model Info
The trained model file efficientnet_b0_best.pth (if included) is required for inference.
If the model file exceeds 100 MB and isn’t included, please download it from your shared source (e.g. Google Drive or Hugging Face) and place it in the project root directory.

📦 Requirements
Dependencies listed in requirements.txt:
streamlit==1.50.0
torch
torchvision
pandas
numpy
pillow
matplotlib
reportlab

🩺 Disclaimer
⚠️ This application is intended for educational and research purposes only.
It is not a medical diagnostic tool. Always consult a qualified medical professional before making clinical decisions.
