import os
from pathlib import Path
import io
import json
import sys

import torch
import torch.nn as nn
from torchvision import models
import timm

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as T

# --------------------
# Configuration
# --------------------
IMG_SIZE = 224
MODEL_FILENAME = 'model_best.pth'

CLASS_NAMES = sorted([
    "Alternaria",
    "Anthracnose",
    "Bacterial_light",
    "Cercospora",
    "Healthy"
])
NUM_CLASSES = len(CLASS_NAMES)

# --------------------
# Model architecture
# --------------------
class CNN_ViT_Fusion(nn.Module):  # NOSONAR
    def __init__(self, num_classes, pretrained=True, freeze_backbones=True):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn.fc = nn.Identity()  # outputs 512-d

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        if hasattr(self.vit, 'heads'):
            self.vit.heads = nn.Identity()

        fused_dim = 512 + 768
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        if freeze_backbones:
            for p in self.cnn.parameters():
                p.requires_grad = False
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        if vit_feat.dim() > 2:
            vit_feat = vit_feat.view(vit_feat.size(0), -1)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.classifier(fused)
        return logits

# --------------------
# Image preprocessing
# --------------------
MODEL_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    MODEL_NORMALIZE
])

def prepare_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(img)
        return image_tensor.unsqueeze(0)  # batch dim
    except Exception as e:
        print(f"Error during image preparation: {e}")
        return None

# --------------------
# Robust path discovery (handles nested zip extraction)
# --------------------
ROOT = Path(__file__).parent.resolve()

def find_first_dir_named(root: Path, name: str, max_search_depth: int = 4):
    try:
        for p in root.rglob(name):
            if p.is_dir():
                rel = p.relative_to(root)
                if len(rel.parts) <= max_search_depth:
                    return p
    except Exception:
        pass
    candidate = root / name
    return candidate

def find_model_file(root: Path, filename: str):
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return p
    except Exception:
        pass
    candidates = [root / filename, root.parent / filename]
    for c in candidates:
        if c.exists():
            return c
    return None

templates_dir = find_first_dir_named(ROOT, "templates")
static_dir = find_first_dir_named(ROOT, "static")
model_path_auto = find_model_file(ROOT, MODEL_FILENAME)

print("Project root:", ROOT)
print("Using templates dir:", templates_dir, "exists:", templates_dir.exists())
print("Using static dir   :", static_dir, "exists:", static_dir.exists())
print("Auto-detected model:", model_path_auto if model_path_auto is not None else "None found")

# --------------------
# Flask app init
# --------------------
app = Flask(
    __name__,
    static_folder=str(static_dir) if static_dir else None,
    template_folder=str(templates_dir) if templates_dir else None
)

print("Jinja search paths (after init):", getattr(app.jinja_loader, "searchpath", None))

try:
    from chatbot_backend.chatbot_routes import chatbot_bp
    app.register_blueprint(chatbot_bp)
    print("Registered chatbot blueprint.")
except Exception as e:
    print("Could not register chatbot blueprint (continuing without it):", e)

CORS(app)

cnn_model = None
device = torch.device("cpu")

# --------------------
# GOOGLE DRIVE MODEL DOWNLOADER
# --------------------
import requests

def download_model_from_gdrive(file_id, destination):
    print("Checking if model file exists at:", destination)
    if os.path.exists(destination):
        print("Model already exists locally. Skipping download.")
        return

    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")
    except Exception as e:
        print("Failed to download model from Google Drive:", e)

GOOGLE_DRIVE_FILE_ID = "1WmlahNX5M2r-oOP20qLg1cU9TQS4NTnj"

download_model_from_gdrive(
    GOOGLE_DRIVE_FILE_ID,
    str(ROOT / MODEL_FILENAME)
)

# --------------------
# Model loading (robust)
# --------------------
def load_single_model(p):
    """Helper to attempt loading a model from a specific path."""
    print(f"Attempting to load model from: {p}")
    state = torch.load(str(p), map_location=device)

    if isinstance(state, nn.Module):
        print("Model loaded (full model object).")
        return state.to(device).eval()

    # If not a full model, assume it's a state_dict
    model_instance = CNN_ViT_Fusion(num_classes=NUM_CLASSES, pretrained=False, freeze_backbones=False).to(device)
    model_instance.load_state_dict(state)
    model_instance.eval()
    print("Model loaded (state_dict).")
    return model_instance

def try_load_model():
    global cnn_model
    loaded = False

    candidate_paths = []
    if model_path_auto:
        candidate_paths.append(model_path_auto)
    candidate_paths += [
        ROOT / MODEL_FILENAME,
        ROOT.parent / MODEL_FILENAME
    ]

    seen = set()
    filtered = []
    for p in candidate_paths:
        try:
            p = Path(p).resolve()
        except Exception:
            p = Path(p)
        if str(p) in seen:
            continue
        seen.add(str(p))
        filtered.append(p)

    for p in filtered:
        if not p.exists():
            print("Model candidate not found:", p)
            continue
        try:
            cnn_model = load_single_model(p)
            loaded = True
            break
        except Exception as e:
            print(f"Failed loading from {p}: {e}")

    if not loaded:
        print("WARNING: Could not locate or load the model file. Ensure model_best.pth exists in project root or a subfolder. Searched candidates:", filtered)

try_load_model()

# --------------------
# FRONTEND routes
# --------------------
@app.route('/')
def root():
    return render_template("login.html")

@app.route('/login')
def login_page():
    return render_template("login.html")

@app.route('/home')
def home_page():
    return render_template("home.html")

@app.route('/upload')
def upload_page():
    return render_template("upload.html")

@app.route('/disease')
def disease_page():
    return render_template("Disease.html")

@app.route('/history')
def history_page():
    return render_template("history.html")

@app.route('/aboutus')
def aboutus_page():
    return render_template("aboutus.html")

@app.route('/test')
def test_route():
    return "TEST OK"

# --------------------
# API route for status
# --------------------
@app.route('/api')
def api_home():
    status = "API Server Running (PyTorch)"
    if not cnn_model:
        status += " (WARNING: Model failed to load)"
    return jsonify({
        "project_status": status,
        "model_loaded": cnn_model is not None,
        "model_framework": "PyTorch (.pth)",
        "image_size_expected": f"{IMG_SIZE}x{IMG_SIZE}",
        "classes_detected": CLASS_NAMES,
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not cnn_model:
        return jsonify({"error": "Model not loaded. Check server logs for details.", "status": "Error"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'.", "status": "Error"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    input_tensor = prepare_image(image_bytes)
    if input_tensor is None:
        return jsonify({"error": "Failed to process the uploaded image file.", "status": "Error"}), 400

    try:
        with torch.no_grad():
            output = cnn_model(input_tensor.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence_score = float(torch.max(probabilities).item())
            predicted_index = int(torch.argmax(probabilities).item())
            predicted_disease = CLASS_NAMES[predicted_index]

            return jsonify({
                "status": "Success",
                "predicted_disease": predicted_disease,
                "confidence_score": f"{confidence_score * 100:.2f}%"
            })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}", "status": "Error"}), 500

# --------------------
# Disease measures endpoint
# (unchanged)
# ...
# --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
