import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from typing import Tuple, Dict

import streamlit as st

import base64
from pathlib import Path
import streamlit as st

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(10, 20, 30, 0.70), rgba(10, 20, 30, 0.86)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

# Call once at the top of your app
set_background("bg_network.png")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ============================================================
# 📄 PDF REPORT GENERATION (using reportlab)
# ============================================================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import io

def generate_pdf_report_dark(pred_name, probs_dict, img, overlay):
    """Generate a sleek, well-aligned dark-theme NeuroScan AI report."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Background ---
    c.setFillColorRGB(0.06, 0.13, 0.15)  # #0F2027
    c.rect(0, 0, width, height, fill=1, stroke=0)

    cyan = colors.Color(0, 0.88, 1)
    white = colors.white
    light_gray = colors.Color(0.7, 0.8, 0.82)

    # --- Header ---
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(cyan)
    c.drawString(60, height - 70, "■ NeuroScan AI - MRI Analysis Report")

    # --- Divider Line ---
    c.setStrokeColor(cyan)
    c.setLineWidth(1.2)
    c.line(60, height - 78, width - 60, height - 78)

    # --- Prediction Section ---
    y_start = height - 120
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(cyan)
    c.drawString(60, y_start, "Prediction Summary:")

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(white)
    c.drawString(60, y_start - 26, f"Predicted Class: {pred_name}")

    # --- Probabilities ---
    y = y_start - 70
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(cyan)
    c.drawString(60, y, "Class Probabilities:")

    y -= 22
    c.setFont("Helvetica", 12)
    for cls, prob in probs_dict.items():
        c.setFillColor(white)
        c.drawString(85, y, f"• {cls:<10} — {prob:.2f}%")
        y -= 18

    # --- Second Divider ---
    c.setStrokeColor(cyan)
    c.setLineWidth(0.8)
    c.line(60, y - 10, width - 60, y - 10)

    # --- Images Section ---
    img_y = 180
    img_size = 220

    def pil_to_reader(pil_img):
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        return ImageReader(img_buf)

    # Labels
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(cyan)
    c.drawCentredString(width / 2 - 100, img_y + img_size + 30, "Uploaded MRI")
    c.drawCentredString(width / 2 + 100, img_y + img_size + 30, "Grad-CAM Overlay")

    # Images (centered)
    c.drawImage(pil_to_reader(img), width / 2 - 210, img_y, width=img_size, height=img_size, mask='auto')
    c.drawImage(pil_to_reader(overlay), width / 2 + 10, img_y, width=img_size, height=img_size, mask='auto')

    # --- Footer Divider ---
    c.setStrokeColor(colors.Color(0.0, 0.6, 0.8))
    c.setLineWidth(0.6)
    c.line(60, 100, width - 60, 100)

    # --- Footer Text ---
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(light_gray)
    footer = (
        "Disclaimer: This report is for research and educational purposes only. "
        "Not for clinical diagnosis. Interpret alongside professional evaluation."
    )
    c.drawCentredString(width / 2, 85, footer)

    # Save PDF
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- HELPER FUNCTIONS FOR ROBUST IMAGE PATHS (CRITICAL FIX) ---
def get_local_image_path(filename):
    """
    Returns the filename. Streamlit can typically serve files in the root.
    If the simple filename fails in the HTML, you might need to try a local path
    or move the files to a dedicated 'static' directory if deploying.
    We return the simple filename, which works in local Streamlit deployments.
    """
    return filename

# --- END HELPER FUNCTIONS ---

# Page config
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

import base64

def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -------------------------------------------------------------------
# APP_CSS: UPDATED STYLING FOR SAMPLE IMAGES (SIZE REDUCTION TO 100px)
# -------------------------------------------------------------------
APP_CSS = """
<style>
/* ==================================================================== */
/* ------------------------ CORE DESIGN SYSTEM ------------------------ */
/* ==================================================================== */

/* Basic resets */
body {-webkit-font-smoothing: antialiased;}

/* SCIMED DARK THEME PALETTE */
:root {
    --primary-brand: #00E0FF; /* Bright Cyan */
    --background-color: #0F2027; /* Deepest Blue/Black */
    --card-background: #203A43; /* Dark Teal/Blue Gray */
    --accent-gradient: linear-gradient(90deg, #00E6FF, #00E660); /* Cyan to Green/Cyan */
    --text-color: #E3F2FD; /* Very Light Blue/White */
    --muted-text: #90A4AE; /* Muted Gray-Blue */
}

/* Streamlit Theme Overrides for Dark Mode */
.stApp {
    background-color: var(--background-color);
    /* The overall background image reference is intentionally removed from pure CSS for reliability */
    background-repeat: repeat;
    background-size: cover;
    background-attachment: fixed;
    color: var(--text-color);
}

/* Header, body, and major component text/color */
h1, h2, h3, h4, h5, h6 { color: var(--primary-brand) !important; }
.small-muted { color: var(--muted-text); font-size:14px; }
p { color: var(--text-color); }

/* Streamlit component backgrounds (e.g., table, dataframes) */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    background-color: var(--card-background) !important;
    border: 1px solid #334D58;
    border-radius: 8px;
}
.stTable > div > div > table > tbody > tr > td {
    color: var(--text-color) !important;
    border-bottom: 1px solid #334D58 !important;
}

/* ==================================================================== */
/* -------------------------- APP COMPONENTS -------------------------- */
/* ==================================================================== */

/* Top bar (small informational strip) */
.top-bar {
    background: var(--card-background); /* Dark Teal/Blue Gray */
    color: var(--text-color);
    padding: 6px 18px;
    border-radius: 8px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.top-bar .left { font-weight:600; }
.top-bar .right { font-size:14px; opacity:0.9; }

/* Header */
.header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-top: 12px;
}

/* Brand styling: CORRECTED to ensure text is white/light */
.brand div { color: var(--text-color) !important; }
.brand div:nth-child(2) div { color: var(--text-color) !important; }
.brand div:nth-child(2) .small-muted { color: var(--muted-text) !important; }

/* Logo */
.brand .logo {
    width:56px;
    height:56px;
    border-radius:10px;
    /* Cyan Gradient for logo */
    background:var(--accent-gradient);
    display:flex;
    align-items:center;
    justify-content:center;
    color: #0F2027; /* Dark text on bright background */
    font-weight:700;
    box-shadow: 0 8px 26px rgba(0,0,0,0.4);
}

/* Nav */
.nav {
    display:flex;
    gap:14px;
    align-items:center;
}
.nav a {
    color: var(--muted-text);
    text-decoration:none;
    font-weight:600;
    padding:8px 12px;
    border-radius:8px;
    transition: color 0.2s, background 0.2s;
}
.nav a:hover {
    color: var(--primary-brand);
}
/* Cyan CTA button */
.nav a.cta {
    background: var(--accent-gradient);
    color: #0F2027 !important; /* Dark text on bright button */
    box-shadow: 0 8px 20px rgba(0, 224, 255, 0.4);
}

/* ==================================================================== */
/* --------------------------- HERO SECTION --------------------------- */
/* ==================================================================== */

/* Hero with background image */
.hero {
    margin-top:20px;
    border-radius: 12px;
    overflow:hidden;
    position:relative;
    display:flex;
    align-items:center;
    min-height: 420px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
/* Reduced opacity overlay */
.hero::after {
    content: "";
    position:absolute;
    inset:0;
    background: linear-gradient(90deg, rgba(15,32,39,0.85) 0%, rgba(15,32,39,0.55) 55%, rgba(15,32,39,0.3) 100%);
}

.hero-content {
    position:relative;
    z-index:2;
    padding:48px;
    /* ADJUSTED WIDTH to give text more space - REMOVING this as we use Streamlit columns */
    /* max-width:62%; */
}
.hero h1 { color:#ffffff !important; font-size:46px; margin:0 0 0 0; }
.hero p { color:var(--text-color) !important; font-size:16px; margin-bottom:18px; }
.hero .cta-hero {
    background: var(--accent-gradient);
    color: #0F2027;
    padding:10px 18px;
    border-radius:10px;
    font-weight:700;
    text-decoration:none;
    display:inline-block;
    transition: transform 0.2s;
}
.hero .cta-hero:hover { transform: translateY(-2px); }

/* Hero-card styling not needed for two columns, but keeping related styles */
.hero-card {
    position: relative;
    z-index: 2;
    /* width: 500px; */ /* Removed fixed width */
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    justify-content: center;
}

/* The actual graphic (the brain in the tube) */
.hero-image-graphic {
    position: absolute;
    top: 50%; /* Center vertically */
    right: 50px;
    transform: translateY(-50%); /* Center vertically */
    width: 420px;
    height: 480px;
    z-index: 2; /* Ensure it's above the background and overlay */
    /* The image URL will be injected via Python */
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}

/* Reposition the small text below the absolute graphic */
.hero-card .small-muted-override {
    position: absolute; /* Changed to absolute to position it */
    z-index: 3;
    bottom: 20px; /* Position at the bottom of the hero section */
    right: 48px; /* Align with padding */
    font-size: 13px;
    color: #6b7c7a;
}

/* ==================================================================== */
/* --------------------------- MAIN CONTENT --------------------------- */
/* ==================================================================== */

/* Features section */
.features {
    margin-top:28px;
    display:grid;
    grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
    gap:18px;
}
.feature-card {
    background:var(--card-background); /* Dark card background */
    border-radius:12px;
    padding:20px;
    text-align:left;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 18px 48px rgba(0,0,0,0.6);
}
.feature-icon {
    width:56px;
    height:56px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(90deg, #334D58, #4D646F);
    color:var(--primary-brand); /* Cyan icon color */
    font-weight:800;
    margin-bottom:12px;
}

/* How it works timeline */
.timeline {
    display:flex;
    gap:14px;
    margin-top:18px;
}
.timeline .step {
    background:var(--card-background); /* Dark step background */
    padding:18px;
    border-radius:12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    flex:1;
    transition: transform 0.28s ease;
}
.step:hover { transform: translateY(-6px); }

/* Predictor card style (main interactive area) */
.predictor {
    background:var(--card-background); /* Darker main card background */
    padding:18px;
    border-radius:12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
/* Highlight prediction text in Cyan */
.predictor h3 span { color:var(--primary-brand) !important; }

/* Grad-cam and images responsive */
.img-side { border-radius:8px; overflow:hidden; }

/* FAQ card */
.faq {
    display:grid;
    grid-template-columns: 1fr 1fr;
    gap:16px;
    margin-top:18px;
}
/* Ensure FAQ cards use the dark background like the rest of the site */
.faq > div {
    background:var(--card-background) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
}

/* Footer */
.footer {
    margin-top:28px;
    padding:18px;
    color:var(--muted-text);
    border-radius:8px;
    background: var(--card-background); /* Dark footer background */
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* Streamlit specific component styling for dark mode */
/* File uploader background */
/* ==== FIX: REMOVE RECTANGULAR UPLOADER BACKGROUND ==== */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stFileUploader"] > div:first-child > div:nth-child(2) {
    background: transparent !important;
    border: 1px dashed var(--primary-brand) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

div[data-testid="stFileUploader"] label {
    color: var(--muted-text) !important;
}

/* ==== REMOVE DARK COLUMN BOX AROUND UPLOADER ==== */
div[data-testid*="column"] {
    background: transparent !important;
    box-shadow: none !important;
    min-height: auto !important;
    padding: 0 !important;
}

/* ************ MODIFIED CSS FOR SLIGHTLY SMALLER SAMPLE IMAGES ************ */
div[data-testid*="column"] { /* Targets the Streamlit column wrapper around images */
    background-color: var(--card-background); /* Use your card background */
    border-radius: 12px; /* Match your card border-radius */
    padding: 10px; /* Add some internal padding */
    box-shadow: 0 4px 10px rgba(0,0,0,0.3); /* Subtle shadow for depth */
    display: flex; /* Use flexbox for centering */
    flex-direction: column;
    align-items: center; /* Centers items horizontally */
    justify-content: center;
    text-align: center;
}

div[data-testid*="column"] img { /* Targets the actual image within the column */
    border: 1px solid #334D58; /* Existing border, slightly refined */
    border-radius: 8px; /* Maintain rounded corners for the image itself */
    background-color: var(--background-color); /* Matches app background */
    object-fit: contain; /* Ensure image scales correctly within its box */
    max-width: 100px; /* <<< --- REDUCED IMAGE SIZE from 120px to 100px */
    width: auto;
    height: auto;
}
/* Style for the caption text under sample images */
div[data-testid*="column"] > div > div > div:last-child {
    color: var(--muted-text);
    font-size: 13px;
    margin-top: 8px; /* Space between image and caption */
}
/* ************************************************************ */

/* Animations (fade-in) */
.fade-in {
    animation: fadeInUp 0.9s ease both;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Small screens */
@media (max-width: 880px) {
    .hero { flex-direction:column; min-height:300px; }
    .hero-card { width:100%; margin-top:12px; }
    .nav { display:none; }
}
</style>
"""
# -------------------------------------------------------------------
# 🔧 ADDITIONAL GLOW + SPACING FIXES
# -------------------------------------------------------------------
APP_CSS += """
<style>
/* === GLOW + TEXT CONSISTENCY FIXES (UNIFIED SIZE + GLOW) === */

/* Make all major section headers consistent in size and glow */
h2, h3, h4 {
    color: #00eaff !important;
    font-size: 1.8rem !important;  /* unified font size for all section headings */
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    text-shadow: 0 0 6px rgba(0, 234, 255, 0.45), 0 0 12px rgba(0, 234, 255, 0.3) !important;
}

/* Specific matching for known section headers */
h3:contains("About NeuroScan AI"),
h3:contains("Sample MRI References"),
h3:contains("FAQ & Contact"),
h3:contains("How NeuroScan AI Works") {
    font-size: 1.8rem !important;
    color: #00eaff !important;
    text-shadow: 0 0 6px rgba(0, 234, 255, 0.45), 0 0 12px rgba(0, 234, 255, 0.3) !important;
}

/* Tighten section spacing so everything feels evenly aligned */
h3 {
    margin-top: 40px !important;
    margin-bottom: 16px !important;
}
/* Tighten spacing before Predictor section */
div[data-testid="stVerticalBlock"] > div:has(#predictor) {
    margin-top: -26px !important;  /* adjust if still slightly large */
    padding-top: 0 !important;
}
/* --- Section Spacing Adjustments --- */

/* Add small bottom space to section above Predictor */
#about {
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.5rem !important;
}

/* Add a small positive space above Predictor heading */
a#predictor + div h3 {
    margin-top: 20px !important;
}

html {
  scroll-behavior: smooth;
}

</style>
"""
# ✅ Apply the updated CSS styles globally
st.markdown(APP_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# ML & Grad-CAM Code (Standardized and Kept as Provided)
# -------------------------------------------------------------------

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


@st.cache_resource(show_spinner=False)
def load_model_checkpoint(path: str, device: torch.device):
    """Load the EfficientNet-B0 model architecture and weights."""
    try:
        model = models.efficientnet_b0(weights=None)
    except Exception:
        model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and ("state_dict" in state or any(k.startswith('module.') or k.startswith('model.') for k in state.keys())):
        state_dict = state.get("state_dict", state)
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "").replace("model.", "")
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model

# Preprocessing transform (ImageNet standardization)
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
])

class GradCAM:
    """Lightweight Grad-CAM implementation."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self.target_layer = self._find_target_layer()
        if self.target_layer is None:
            raise ValueError("No convolutional layer found in model to use for Grad-CAM.")
        self._register_hooks()

    def _find_target_layer(self):
        target = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target = module
                break
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handle_forward = self.target_layer.register_forward_hook(forward_hook)
        self.handle_backward = self.target_layer.register_backward_hook(backward_hook)

    def remove_hooks(self):
        self.handle_forward.remove()
        self.handle_backward.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations
        if grads is None or activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        weighted_acts = weights * activations
        cam = torch.sum(weighted_acts, dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        cam_np = cam.cpu().numpy()
        cam_resized = np.uint8(255 * cam_np)
        return cam_resized


def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    """Apply preprocessing and return batched tensor on device"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    tensor = PREPROCESS(pil_img).unsqueeze(0)
    return tensor


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap_arr: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlays the heatmap onto the base image."""
    heatmap_img = Image.fromarray(heatmap_arr).resize(pil_img.size, resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_img)
    colormap = mplcm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_np / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)
    base = np.array(pil_img).astype(np.uint8)
    overlay = np.uint8(base * (1 - alpha) + heatmap_colored * alpha)
    return Image.fromarray(overlay)


@st.cache_resource(show_spinner=False)
def get_model_and_cam(path="efficientnet_b0_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(path, device)
    cam = GradCAM(model=model, device=device)
    return model, cam, device


def predict_with_gradcam(pil_image: Image.Image, model: nn.Module, cam_obj: GradCAM, device: torch.device) -> Tuple[str, Dict[str, float], Image.Image, int]:
    """Returns: predicted class name, dict of probs, grad-cam overlay PIL image, predicted index"""
    input_tensor = prepare_image(pil_image).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    probs_dict = {cls: float(round(prob * 100, 4)) for cls, prob in zip(CLASS_NAMES, probs)}
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    try:
        model.zero_grad()
        input_tensor_for_cam = prepare_image(pil_image).to(device)
        _ = model(input_tensor_for_cam)
        heatmap_uint8 = cam_obj(input_tensor_for_cam, class_idx=pred_idx)
    except Exception:
        try:
            input_tensor_for_cam = prepare_image(pil_image).to(device)
            h = input_tensor_for_cam.shape[2]
            w = input_tensor_for_cam.shape[3]
        except Exception:
            h, w = 224, 224
        heatmap_uint8 = np.zeros((h, w), dtype=np.uint8)

    overlay = overlay_heatmap_on_image(pil_image.convert("RGB"), heatmap_uint8, alpha=0.5)
    return pred_name, probs_dict, overlay, pred_idx

# -------------------------------------------------------------------
# App content & layout (HTML/Streamlit UI)
# -------------------------------------------------------------------

# Top informational bar
with st.container():
    st.markdown(
        """
        <div class="top-bar fade-in" style="animation-delay: 0.1s; padding: 6px 18px;">
            <div class="left" style="font-weight:700; font-size:15px; color:#E3F2FD;">
                NeuroScan AI: <span style="color:#00E0FF;">Explainable AI for Brain MRI Screening</span>
            </div>
            <div class="right" style="font-size:13px; opacity:0.9;">
                Classifies: Glioma · Meningioma · Pituitary · No Tumor
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Header (brand + nav)
with st.container():
    st.markdown(
        """
        <div class="header fade-in">
            <div class="brand">
                <div class="logo">NS</div>
                <div>
                    <div style="font-size:18px;font-weight:800;">NeuroScan AI</div>
                    <div class="small-muted">AI-assisted Brain MRI Screening</div>
                </div>
            </div>
            <div class="nav">
                <a href="#home">Home</a>
                <a href="#about">About</a>
                <a href="#how">How it Works</a>
                <a href="#predictor" class="cta">Predictor</a>
                <a href="#contact">Contact</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Image paths (CONFIRMED .PNG EXTENSIONS)
LOCAL_HERO_FNAME = "hero_brain_tube.jpg" # Using .jpg from prompt
LOCAL_BG_FNAME = "bg_network.png"       # CONFIRMED .png
LOCAL_SAMPLE_1 = "Axial.png"
LOCAL_SAMPLE_2 = "Brainscan.png"
LOCAL_SAMPLE_3 = "Radiology.png"

# Fallback/Remote URLs (not affected by your local files)
HERO_BG_REMOTE = "https://images.unsplash.com/photo-1586773860418-11b4efb4a43a?q=80&w=2080&auto=format&fit=crop&ixlib=rb-4.0.3&s=6d8c6aef6c1b6b1c9cb56e7b2b1b9b4b"
HERO_IMAGE_REMOTE = "https://images.unsplash.com/photo-1602491699113-8f8f6b0d6f2d?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=26a2a3f6ddf0f6c3b9b1e2a3f7d6c0b9"

# Determine final URLs
HERO_BG_URL = get_local_image_path(LOCAL_BG_FNAME) if os.path.exists(LOCAL_BG_FNAME) else HERO_BG_REMOTE
HERO_IMAGE_URL = get_local_image_path(LOCAL_HERO_FNAME) # Use the uploaded brain image URL


# Hero Section - MODIFIED TO INCLUDE IMAGE 1 ON THE RIGHT
# --- HOME SECTION ---
st.markdown('<a id="home"></a>', unsafe_allow_html=True)
# --- HERO SECTION (working image with base64) ---
hero_base64 = get_base64_image("hero_brain_tube.png")

st.markdown(f"""
<style>
@keyframes floatBrain {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
    100% {{ transform: translateY(0px); }}
}}

.hero-section {{
    background: rgba(255, 255, 255, 0.005);  /* ⬅️ ultra transparent */
    backdrop-filter: blur(2px) saturate(120%);
    border: 1px solid rgba(255,255,255,0.02);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 70px 60px;
    border-radius: 18px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}

.hero-section::before {{
    content: "";
    position: absolute;
    right: 10%;
    top: 10%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle at center, rgba(0,255,255,0.12), transparent 80%);
    filter: blur(140px);
    z-index: 0;
}}

.hero-left {{
    flex: 1;
    z-index: 2;
    max-width: 52%;
}}

.hero-left h1 {{
    color:#00E0FF;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 16px;
    text-shadow: 0 0 20px rgba(0,255,255,0.2);
}}

.hero-left p {{
    color:#E3F2FD;
    font-size: 17px;
    line-height: 1.6;
    margin-bottom: 22px;
    opacity: 0.9;
}}

.hero-right {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
}}

.hero-right img {{
    width: 85%;
    max-width: 360px;
    border-radius: 16px;
    animation: floatBrain 6s ease-in-out infinite;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    transition: transform 0.4s ease, box-shadow 0.3s ease;
}}

.hero-right img:hover {{
    transform: scale(1.03) translateY(-5px);
    box-shadow: 0 0 45px rgba(0,255,255,0.4);
}}

.cta-hero {{
    background: linear-gradient(90deg, #00E0FF, #00FFA3);
    color:#0F2027 !important;
    padding:12px 22px;
    border-radius: 10px;
    font-weight:700;
    text-decoration:none;
    box-shadow: 0 0 12px rgba(0,255,255,0.25);
    transition: all 0.3s ease;
}}

.cta-hero:hover {{
    background: linear-gradient(90deg, #00FFA3, #00E0FF);
    box-shadow: 0 0 25px rgba(0,255,255,0.4);
}}
</style>

<div class="hero-section">
    <div class="hero-left">
        <h1>Welcome to <br>NeuroScan AI</h1>
        <p>
            Advanced, explainable brain MRI analysis using deep learning.<br>
            NeuroScan AI classifies MRI scans into Glioma, Meningioma, Pituitary, or No Tumor — 
            and provides <b>Grad-CAM</b> visual explanations to highlight regions that influenced the decision.
        </p>
        <a class="cta-hero" href="#predictor">Get Started — Upload MRI</a>
        <div style="margin-top: 14px; color:#B0BEC5;">
            Designed for clinicians & researchers — use as an assistive tool.
        </div>
    </div>
    <div class="hero-right">
        <img src="data:image/png;base64,{hero_base64}" alt="NeuroScan Brain">
    </div>
</div>
""", unsafe_allow_html=True)

# Features section
with st.container():
    st.markdown("<div style='margin-top:22px;'><h3>Key Features</h3></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="features fade-in">
            <div class="feature-card">
                <div class="feature-icon">⏱️</div>
                <div style="font-weight:700;">Rapid Inference</div>
                <div class="small-muted">Quick results to help triage cases efficiently.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div style="font-weight:700;">4-class Brain Tumor Classification</div>
                <div class="small-muted">Glioma, Meningioma, Pituitary, No Tumor.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div style="font-weight:700;">Explainability (Grad-CAM)</div>
                <div class="small-muted">Visual heatmaps highlight regions influencing predictions.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div style="font-weight:700;">Detailed Probabilities</div>
                <div class="small-muted">Per-class confidence scores for informed decisions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- HOW IT WORKS SECTION ---
st.markdown('<a id="how"></a>', unsafe_allow_html=True)
with st.container():
    st.markdown("<div style='margin-top:26px;'><h3>How NeuroScan AI Works</h3></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="timeline fade-in">
            <div class="step">
                <div style="font-weight:800; font-size:18px;">1. Upload MRI</div>
                <div class="small-muted">Upload a single axial/coronal MRI slice as JPG/PNG.</div>
            </div>
            <div class="step">
                <div style="font-weight:800; font-size:18px;">2. Model Inference</div>
                <div class="small-muted">An EfficientNet-based model predicts class probabilities.</div>
            </div>
            <div class="step">
                <div style="font-weight:800; font-size:18px;">3. Grad-CAM</div>
                <div class="small-muted">Grad-CAM highlights important regions used by the network.</div>
            </div>
            <div class="step">
                <div style="font-weight:800; font-size:18px;">4. Review & Confirm</div>
                <div class="small-muted">Clinician reviews results and integrates with clinical context.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# About / Explanation
# --- ABOUT SECTION ---
st.markdown('<a id="about"></a>', unsafe_allow_html=True)
# ---------- ABOUT SECTION (replace your current About block with this) ----------
from pathlib import Path
import base64

# helper: return base64 for a file if exists, otherwise None
def _get_base64_if_exists(path: str):
    if Path(path).exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Try local workflow image; fallback to a small remote icon (png)
local_workflow = "workflow.png"
workflow_b64 = _get_base64_if_exists(local_workflow)
fallback_icon = "https://upload.wikimedia.org/wikipedia/commons/8/83/VisualEditor_-_Icon_-_Flowchart.svg"

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='color:var(--primary-brand);'>About NeuroScan AI</h2>", unsafe_allow_html=True)

if workflow_b64 is None:
    # Use remote image URL in markup
    workflow_img_tag = f'<img src="{fallback_icon}" style="width:160px; border-radius:14px; box-shadow:0 0 22px rgba(0,255,255,0.18);"/>'
else:
    workflow_img_tag = f'<img src="data:image/png;base64,{workflow_b64}" style="width:160px; border-radius:14px; box-shadow:0 0 22px rgba(0,255,255,0.18);"/>'

st.markdown(f"""
<style>
@keyframes floatWorkflow {{
  0% {{ transform: translateY(0px); }}
  50% {{ transform: translateY(-6px); }}
  100% {{ transform: translateY(0px); }}
}}
/* Glass panel */
.about-glass {{
    display:flex;
    gap:28px;
    align-items:center;
    justify-content:space-between;
    background: var(--card-background); /* Updated: use card background */
    border-radius:18px;
    padding:28px 32px;
    box-shadow: 0 12px 40px rgba(2,18,24,0.40);
    backdrop-filter: blur(8px) saturate(130%);
    border: 1px solid rgba(255,255,255,0.03);
    margin-bottom:28px;
}}
.about-text {{
    flex: 2;
    color: #E3F2FD;
    font-size:17px;
    line-height:1.65;
}}
.about-text b {{
    font-weight:700;
}}
.about-image {{
    flex: 0 0 180px;
    text-align:center;
}}
.about-image img {{
    animation: floatWorkflow 6s ease-in-out infinite;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
    box-shadow:
        0 0 25px rgba(0,255,255,0.45),
        0 0 70px rgba(0,255,255,0.3),
        inset 0 0 15px rgba(0,255,255,0.25);
    border-radius:16px;
    background: rgba(0, 255, 255, 0.06);
    backdrop-filter: blur(12px) saturate(140%);
    border-radius: 18px;
    padding: 12px;
}}

.about-image img:hover {{
    transform: scale(1.05);
    box-shadow:
        0 0 40px rgba(0,255,255,0.6),
        0 0 100px rgba(0,255,255,0.4),
        inset 0 0 20px rgba(0,255,255,0.3);
}}

.about-caption {{
    margin-top:10px;
    color: #BFD8DE;
    font-size:14px;
}}
@media (max-width: 880px) {{
    .about-glass {{ flex-direction:column; gap:18px; }}
    .about-image {{ margin-top:6px; }}
}}
</style>

<div class="about-glass">
  <div class="about-text">
    <div style="font-size:18px;"><b>Purpose:</b> Empowering Clinical Insight with AI</div>
    <div style="height:14px;"></div>
    <div style="color:#C8DCE0;">
        Our model delivers fast, explainable MRI classification for common brain tumor types, helping clinicians make informed decisions with confidence. Trained on a curated dataset of MRI slices, it offers interpretable results designed to complement—not replace—clinical expertise. Use it as a diagnostic aid to streamline workflows and enhance patient care.
    </div>
  </div>

  <div class="about-image">
    {workflow_img_tag}
    <div class="about-caption">Clinical workflows integrated</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- PREDICTOR SECTION ----------
# --- PREDICTOR SECTION ---
st.markdown('<a id="predictor"></a>', unsafe_allow_html=True)
# ---- FINAL FIX FOR PREDICTOR SECTION ----
st.markdown("""
<style>

/* ✨ Predictor Container */
.predictor {
    background: rgba(8, 25, 35, 0.6);
    border-radius: 24px;
    padding: 48px;
    margin-top: 0 !important;
    box-shadow: 0 0 45px rgba(0,255,255,0.1),
                inset 0 0 25px rgba(0,255,255,0.05);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid rgba(0,255,255,0.12);
    transition: all 0.3s ease-in-out;
}
.predictor:hover {
    box-shadow: 0 0 60px rgba(0,255,255,0.25),
                0 0 100px rgba(0,150,255,0.18);
}

/* 💠 Section Title */
h3 {
    color: #00E0FF !important;
    text-shadow: 0 0 18px rgba(0,255,255,0.5);
    font-size: 1.9rem !important;
    margin-bottom: 0.5rem;
}

/* 💧 Enhanced File Uploader Styling (Full visibility fix) */
[data-testid="stFileUploader"] section {
    background: rgba(10, 25, 35, 0.7) !important;
    border: 1px solid rgba(0,255,255,0.25) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 25px rgba(0,255,255,0.3),
                inset 0 0 20px rgba(0,200,255,0.08);
    transition: 0.3s ease-in-out;
}

[data-testid="stFileUploader"] section:hover {
    background: rgba(12, 35, 45, 0.9) !important;
    box-shadow: 0 0 45px rgba(0,255,255,0.4),
                inset 0 0 25px rgba(0,200,255,0.15);
}

/* ✅ Make ALL uploader text clearly visible */
[data-testid="stFileUploader"] div[role="button"],
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label {
    color: #AFFFFF !important;
    text-shadow: 0 0 12px rgba(0,255,255,0.6);
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* 📁 Glowing upload icon */
[data-testid="stFileUploader"] svg {
    filter: drop-shadow(0 0 14px rgba(0,255,255,0.75));
    stroke: #00FFFF !important;
    stroke-width: 2 !important;
}


/* 🔹 Scanner Glow Hover Effect */
[data-testid="stFileUploader"] section:hover {
    box-shadow: 0 0 45px rgba(0,255,255,0.35),
                0 0 15px rgba(0,200,255,0.25),
                inset 0 0 25px rgba(0,255,255,0.15);
    border-color: rgba(0,255,255,0.5) !important;
}

/* 🔹 Moving Glow Line Effect */
[data-testid="stFileUploader"] section::before {
    content: "";
    position: absolute;
    top: -150%;
    left: 0;
    width: 100%;
    height: 300%;
    background: linear-gradient(180deg,
        rgba(0,255,255,0) 0%,
        rgba(0,255,255,0.2) 50%,
        rgba(0,255,255,0) 100%);
    opacity: 0;
    transition: opacity 0.4s ease;
}
[data-testid="stFileUploader"] section:hover::before {
    opacity: 1;
    animation: scannerMove 3s linear infinite;
}
@keyframes scannerMove {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}

/* 💎 Upload Text – More Visible + Glow */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div[role="button"],
[data-testid="stFileUploader"] p {
    color: #E3FAFF !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    text-shadow:
        0 0 8px rgba(0,255,255,0.45),
        0 0 18px rgba(0,255,255,0.25);
    transition: all 0.3s ease-in-out;
}

/* On Hover: Make text even brighter */
[data-testid="stFileUploader"] section:hover label,
[data-testid="stFileUploader"] section:hover div[role="button"],
[data-testid="stFileUploader"] section:hover p {
    color: #FFFFFF !important;
    text-shadow:
        0 0 10px rgba(0,255,255,0.7),
        0 0 25px rgba(0,255,255,0.5);
}

/* 🔹 Upload Icon Glow */
@keyframes pulseGlow {
    0% { filter: drop-shadow(0 0 3px rgba(0,255,255,0.2)); }
    50% { filter: drop-shadow(0 0 12px rgba(0,255,255,0.8)); }
    100% { filter: drop-shadow(0 0 3px rgba(0,255,255,0.2)); }
}
[data-testid="stFileUploader"] svg {
    animation: pulseGlow 3s ease-in-out infinite;
}

/* 🔹 Browse Files Button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #00D4FF 0%, #00AEEF 100%) !important;
    color: #002830 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,255,255,0.3) !important;
    box-shadow: 0 0 20px rgba(0,255,255,0.25);
    transition: all 0.25s ease-in-out;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(90deg, #00FFFF 0%, #00D0FF 100%) !important;
    color: #001820 !important;
    box-shadow: 0 0 40px rgba(0,255,255,0.6);
    transform: scale(1.05);
}

/* 📁 Icon Glow */
[data-testid="stFileUploader"] svg {
    filter: drop-shadow(0 0 12px rgba(0,255,255,0.5));
}

/* 💙 Predict & Explain Button - Matches 'Browse Files' style */
div.stButton > button[kind="primary"],
div.stButton > button,
button.st-emotion-cache-7ym5gk.ef3psqc12 {
    background: linear-gradient(90deg, #00C6FF 0%, #00BFFF 100%) !important;
    color: #000000 !important; /* Black text */
    font-weight: 800 !important; /* Bolder */
    font-size: 18px !important; /* Larger font */
    border-radius: 12px !important;
    box-shadow: 0 0 20px rgba(0, 191, 255, 0.35);
    padding: 12px 28px !important;
    border: none !important;
    transition: all 0.3s ease-in-out !important;
    text-shadow: none !important;
    letter-spacing: 0.3px !important;
}

/* Hover Effect */
div.stButton > button[kind="primary"]:hover,
div.stButton > button:hover,
button.st-emotion-cache-7ym5gk.ef3psqc12:hover {
    transform: scale(1.06);
    background: linear-gradient(90deg, #00BFFF 0%, #00C6FF 100%) !important;
    color: #000000 !important;
    box-shadow: 0 0 35px rgba(0, 191, 255, 0.6);
}

/* Remove the emoji entirely */
div.stButton > button::before {
    content: "" !important;
}

/* 📊 Chart Dark Mode */
[data-testid="stPlotlyChart"], .stAltairChart, [data-testid="stTable"] {
    background: rgba(6, 20, 30, 0.75) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    padding: 10px;
}

/* 📄 Download PDF Button (Teal-Glow Style) */
div.stDownloadButton > button {
    background: linear-gradient(90deg, #00C6FF 0%, #00BFFF 100%) !important;
    color: #000000 !important;  /* Black text */
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 0 20px rgba(0, 191, 255, 0.35);
    padding: 12px 60px !important;  /* Wider padding for long rectangular shape */
    border: none !important;
    transition: all 0.3s ease-in-out !important;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100% !important;  /* Full width for long rectangular button */
    text-align: center !important;
}

div.stDownloadButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #00BFFF 0%, #00C6FF 100%) !important;
    box-shadow: 0 0 40px rgba(0, 191, 255, 0.6);
}

/* 🧠 Hide model status */
div:has(> .stSuccess) {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- PREDICTOR SECTION ----------------------------------------
st.markdown('<a id="predictor"></a>', unsafe_allow_html=True)

# Compact Predictor container
with st.container():
    # Remove all top/bottom space
    st.markdown("<div style='padding:0; margin:0;'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-top:1.5rem; margin-bottom:0.5rem;'>Predictor</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='margin-top:0; margin-bottom:1rem;'>Upload a brain MRI (.jpg/.png). "
        "The model will return prediction, probabilities, and a Grad-CAM heatmap.</p>",
        unsafe_allow_html=True
    )

    # --- 1️⃣ Predictor (Always visible) ---
    uploaded_file = st.file_uploader(
        "Upload MRI image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Prefer axial brain MRI slices. Single images only."
    )

    # --- Load model once ---
    try:
        model, cam_obj, DEVICE = get_model_and_cam("efficientnet_b0_best.pth")
        st.success("Model ready on CPU")
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        model, cam_obj, DEVICE = None, None, torch.device("cpu")

    # --- 2️⃣ Analyze Section (Visible only after upload) ---
    if uploaded_file is not None:
        try:
            img = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception:
            st.error("Could not read image. Try another file.")
            img = None

        if img is not None:
            st.markdown("<h4 style='color:#00E0FF;'>Analyze</h4>", unsafe_allow_html=True)
            alpha = st.slider("Grad-CAM blend (alpha)", 0.0, 1.0, 0.5, 0.05)
            show_chart = st.checkbox("Show probability bar chart", value=True)
            predict_btn = st.button("Predict & Explain")

            # --- 3️⃣ Prediction Logic ---
            if predict_btn:
                if model is None:
                    st.error("Model unavailable. Please check configuration.")
                else:
                    with st.spinner("Running prediction..."):
                        try:
                            pred_name, probs_dict, overlay_default, pred_idx = predict_with_gradcam(
                                img, model, cam_obj, DEVICE
                            )

                            # Recompute Grad-CAM with slider alpha
                            input_tensor = prepare_image(img).to(DEVICE)
                            model.zero_grad()
                            _ = model(input_tensor)
                            heatmap_uint8 = cam_obj(input_tensor, class_idx=pred_idx)
                            overlay = overlay_heatmap_on_image(img.convert("RGB"), heatmap_uint8, alpha=alpha)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            pred_name, probs_dict, overlay = None, None, None

                    if pred_name:
                        # --- Prediction Output ---
                        st.markdown(
                            f"<h4 style='color:#00E0FF;'>Prediction:</h4><p style='font-size:22px;font-weight:800;color:#00E0FF;'>{pred_name}</p>",
                            unsafe_allow_html=True
                        )

                        # --- MRI + Grad-CAM Side-by-Side ---
                        st.markdown("<h4 style='color:#00E0FF;'>Visualization</h4>", unsafe_allow_html=True)
                        col_img1, col_img2 = st.columns(2, gap="large")

                        with col_img1:
                            st.markdown("<h5 style='color:#00E0FF;'>Uploaded MRI</h5>", unsafe_allow_html=True)
                            st.image(img, use_container_width=True, output_format="PNG")

                        with col_img2:
                            st.markdown("<h5 style='color:#00E0FF;'>Grad-CAM Overlay</h5>", unsafe_allow_html=True)
                            st.image(overlay, use_container_width=True, output_format="PNG")

                        # --- Probability Table + Chart ---
                        prob_df = pd.DataFrame({
                            "Class": list(probs_dict.keys()),
                            "Probability (%)": list(probs_dict.values())
                        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

                        st.markdown("<h4 style='color:#00E0FF;margin-top:2rem;'>Prediction Confidence</h4>", unsafe_allow_html=True)
                        col_table, col_chart = st.columns(2, gap="large")

                        with col_table:
                            st.markdown("**Probability Table**")
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)

                        with col_chart:
                            if show_chart:
                                st.markdown("**Probability Distribution**")
                                st.bar_chart(prob_df.set_index("Class"))

                        st.caption(
                            "💡 *Warmer regions in Grad-CAM indicate higher model focus. Use results alongside expert medical evaluation.*"
                        )
                        # --- PDF Report Download (with Grad-CAM + MRI) ---
                        pdf_buffer = generate_pdf_report_dark(pred_name, probs_dict, img, overlay)
                        st.download_button(
                            label="📄 Download Report (PDF)",
                            data=pdf_buffer,
                            file_name=f"NeuroScanAI_Report_{pred_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
    else:
        st.info("Please upload a brain MRI image to begin analysis.")

    # --- 4️⃣ Sample MRI References (Always visible at bottom) ---
    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#00E0FF; margin-top: 0.5rem;'>Sample MRI References</h4>", unsafe_allow_html=True)
    sample_imgs = [
        ("Axial MRI sample", LOCAL_SAMPLE_1 if os.path.exists(LOCAL_SAMPLE_1) else "https://raw.githubusercontent.com/streamlit/example-data/main/brain1.jpg"),
        ("Brain scan (MRI)", LOCAL_SAMPLE_2 if os.path.exists(LOCAL_SAMPLE_2) else "https://raw.githubusercontent.com/streamlit/example-data/main/brain2.jpg"),
        ("Radiology view", LOCAL_SAMPLE_3 if os.path.exists(LOCAL_SAMPLE_3) else "https://raw.githubusercontent.com/streamlit/example-data/main/brain3.jpg")
    ]
    cols = st.columns(len(sample_imgs))
    for (title, url), c in zip(sample_imgs, cols):
        with c:
            st.image(url, caption=title, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# Remaining content (FAQ and Footer)
# -------------------------------------------------------------------
# FAQ and Footer (based on image 3 content)
# --- CONTACT / FAQ SECTION ---
st.markdown('<a id="contact"></a>', unsafe_allow_html=True)
# (FAQ & Contact section follows)
# -------------------------------------------------------------------
# reduce vertical space before FAQ
st.markdown("""
<h3 id='faq' style='margin-top:12px;'>FAQ & Contact</h3>
<div class='faq'>
    <div style='padding:18px;border-radius:12px;'>
        <h4>Is NeuroScan AI a diagnostic tool?</h4>
        <div class='small-muted'>No - it is an assistive tool to highlight possible findings. Diagnosis must be confirmed by a clinician.</div>
    </div>
    <div style='padding:18px;border-radius:12px;'>
        <h4>What image quality is required?</h4>
        <div class='small-muted'>Clear MRI slices (axial/coronal), minimal motion artifact, preferably preprocessed to similar contrast to training data.</div>
    </div>
    <div style='padding:18px;border-radius:12px;'>
        <h4>Contact & Support</h4>
        <div class='small-muted'>Email: contact@neuroscan.ai<br>Phone: +1 (555) 234-5678</div>
    </div>
    <div style='padding:18px;border-radius:12px;'>
        <h4>Research use</h4>
        <div class='small-muted'>If using for research, document dataset and model version and obtain necessary approvals.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer fade-in">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-weight:800;font-size:16px;">© NeuroScan AI • 2025 • For educational & research use</div>
        </div>
        <div class="small-muted">Built with 💙 • Explainability: Grad-CAM • Model: EfficientNet-B0</div>
    </div>
    """,
    unsafe_allow_html=True
)
