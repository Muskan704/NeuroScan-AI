🧠 NeuroScan AI — Brain Tumor Detection using EfficientNet-B0
NeuroScan AI is an advanced medical imaging web application built with Streamlit, designed to assist clinicians and researchers in brain tumor detection using MRI scans.
It provides classification into four categories — Glioma, Meningioma, Pituitary, and No Tumor — with explainability via Grad-CAM heatmaps.

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
# For Windows
python -m venv venv
venv\Scripts\activate
# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

⚠️ Note: You must activate the virtual environment before installing dependencies, otherwise the app may fail to run due to version conflicts.
🧭 Step 3 — Install Dependencies
pip install -r requirements.txt
🧭 Step 4 — Run the Streamlit App
streamlit run neuroscan_app.py
The app will start at:
👉 http://localhost:8501

🧩 Model Info
The trained model file efficientnet_b0_best.pth (if included) is required for inference.
If the model file exceeds 100MB and isn’t included here, please download it from your shared link (e.g., Google Drive or Hugging Face) and place it in the project root directory.

📦 Requirements
The dependencies are listed in requirements.txt:
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
