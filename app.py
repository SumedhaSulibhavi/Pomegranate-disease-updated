# app.py — cleaned, robust, and tolerant to nested/zip extraction layouts

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
class CNN_ViT_Fusion(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbones=True):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn.fc = nn.Identity()  # outputs 512-d

        # vit_base_patch16_224 -> returns 768-d
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
    """
    Search root and its subfolders (up to a reasonable depth) for the first directory named `name`.
    Falls back to root/name if none found.
    """
    try:
        # rglob will traverse; limit by checking path.parts length to avoid full FS scan
        for p in root.rglob(name):
            if p.is_dir():
                # simple depth control: ensure p is inside the root tree and not too deep
                rel = p.relative_to(root)
                if len(rel.parts) <= max_search_depth:
                    return p
    except Exception:
        pass
    candidate = root / name
    return candidate

def find_model_file(root: Path, filename: str):
    """
    Search for a model file with given filename in root and its subfolders (reasonable depth).
    """
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return p
    except Exception:
        pass
    # fallback to common places
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

# Print Jinja search paths so we can debug TemplateNotFound
print("Jinja search paths (after init):", getattr(app.jinja_loader, "searchpath", None))

# Try to import your blueprint if present (non-fatal)
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
# FRONTEND routes
# --------------------
@app.route('/')
def root():
    # use render_template which will look into template_folder
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

# --------------------
# Prediction endpoint
# --------------------
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
# --------------------
DISEASE_DATA = {
    "alternaria": {
        "en": {
            "precautions": "Rotate crops and use disease-free seeds. Avoid overhead irrigation and ensure good field drainage.",
            "points": [
                "Apply Difenoconazole or Azoxystrobin fungicides at early signs.",
                "Remove infected leaves and debris immediately.",
                "Use protective fungicidal sprays during high humidity."
            ]
        },
        "kn": {
            "precautions": "ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ ಮತ್ತು ರೋಗಮುಕ್ತ ಬೀಜಗಳನ್ನು ಬಳಸಿ. ಮೇಲಿನ ನೀರಾವರಿ ತಪ್ಪಿಸಿ ಮತ್ತು ಉತ್ತಮ ನೀರು ಹರಿವು ಒದಗಿಸಿ.",
            "points": [
                "ಪ್ರಾರಂಭಿಕ ಲಕ್ಷಣಗಳಾಗುತ್ತಿದ್ದಂತೆ ಡೈಫೆನೊಕನಜೋಲ್ ಅಥವಾ ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ ಬಳಸಿ.",
                "ಸೋಂಕಿತ ಎಲೆಗಳು ಮತ್ತು ಅವಶೇಷಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದುಹಾಕಿ.",
                "ಹೆಚ್ಚಿನ ತೇವಾಂಶದ ಸಮಯದಲ್ಲಿ ರಕ್ಷಣಾತ್ಮಕ ಫಂಗಿಸೈಡ್ ಸಿಂಪಡಿಸಿ."
            ]
        }
    },
    "anthracnose": {
        "en": {
            "precautions": "Use resistant varieties, prune lower branches for air circulation, and sanitize tools regularly.",
            "points": [
                "Treat with Copper Oxychloride after fruit set.",
                "Apply Mancozeb during rainy periods.",
                "Remove and destroy cankers and infected fruits."
            ]
        },
        "kn": {
            "precautions": "ರೋಗ ನಿರೋಧಕ ಜಾತಿಗಳನ್ನು ಬಳಸಿ, ಗಾಳಿಯ ಹರಿವಿಗೆ ಕೆಳಗಿನ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ, ಮತ್ತು ಉಪಕರಣಗಳನ್ನು ಶುದ್ಧಗೊಳಿಸಿ.",
            "points": [
                "ಹಣ್ಣು ಬೆಳವಣಿಗೆ ನಂತರ ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಬಳಸಿ.",
                "ಮಳೆಯ ಸಮಯದಲ್ಲಿ ಮ್ಯಾನ್ಕೋಜೆಬ್ ಸಿಂಪಡಿಸಿ.",
                "ಸೋಂಕಿತ ಕೊಂಬೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
            ]
        }
    },
    "bacterial_blight": {
        "en": {
            "precautions": "Avoid planting in infected soil and disinfect tools. Use certified disease-free planting material.",
            "points": [
                "Apply Streptomycin or Copper Oxychloride.",
                "Prune infected branches well below symptoms.",
                "Avoid working in fields when leaves are wet."
            ]
        },
        "kn": {
            "precautions": "ಸೋಂಕಿತ ಮಣ್ಣಿನಲ್ಲಿ ಹೊಲ ಹಾಕಬೇಡಿ, ಉಪಕರಣಗಳನ್ನು ಶುದ್ಧಪಡಿಸಿ, ರೋಗಮುಕ್ತ ಸಸ್ಯ ವಸ್ತು ಬಳಸಿ.",
            "points": [
                "ಸ್ಟ್ರೆಪ್ಟೊಮೈಸಿನ್ ಅಥವಾ ತಾಮ್ರ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಬಳಸಿ.",
                "ಸೋಂಕಿನ ಲಕ್ಷಣಗಳಿಗಿಂತ ಕೆಳಗೆ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ.",
                "ಎಲೆಗಳು ತೇವವಾಗಿರುವಾಗ ಹೊಲದಲ್ಲಿ ಕೆಲಸ ತಪ್ಪಿಸಿ."
            ]
        }
    },
    "cercospora": {
        "en": {
            "precautions": "Ensure wide spacing between plants. Avoid nitrogen excess.",
            "points": [
                "Use Chlorothalonil or Tebuconazole fungicides.",
                "Improve airflow by pruning excess leaves.",
                "Remove fallen infected leaves."
            ]
        },
        "kn": {
            "precautions": "ಸಸ್ಯಗಳ ನಡುವೆ ಅಗಲ ಅಂತರ ಇರಲಿ. ಹೆಚ್ಚುವರಿ ನೈಟ್ರೋಜನ್ ಬಳಕೆ ತಪ್ಪಿಸಿ.",
            "points": [
                "ಕ್ಲೋರೋಥಾಲೊನಿಲ್ ಅಥವಾ ಟೆಬುಕೊನಜೋಲ್ ಬಳಸಿ.",
                "ಹೆಚ್ಚಿನ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಗಾಳಿಯ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ.",
                "ಬಿದ್ದಿರುವ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
            ]
        }
    },
    "healthy": {
        "en": {
            "precautions": "Continue monitoring and ensure proper nutrition and irrigation.",
            "points": [
                "Maintain soil nutrients and pH.",
                "Prune for sunlight penetration.",
                "Monitor weekly for pests and disease."
            ]
        },
        "kn": {
            "precautions": "ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಸರಿಯಾದ ಪೋಷಕಾಂಶ, ನೀರಾವರಿ ಒದಗಿಸಿ.",
            "points": [
                "ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ ಹಾಗೂ pH ಕಾಪಾಡಿ.",
                "ಸೂರ್ಯಕಿರಣ ಪ್ರವೇಶಕ್ಕೆ ಪ್ರೂನಿಂಗ್ ಮಾಡಿ.",
                "ವಾರಕ್ಕೆ ಒಂದಿಹೋರೊಮ್ಮೆ ಕೀಟ-ರೋಗ ಪರಿಶೀಲಿಸಿ."
            ]
        }
    }
}

@app.route("/measures/<disease_name>", methods=["GET"])
def get_measures(disease_name):
    from flask import request, jsonify
    disease = disease_name.lower()
    lang = request.args.get("lang", "en").lower()

    if disease not in DISEASE_DATA:
        return jsonify({"error": f"Disease details not found for: {disease}"}), 404

    disease_info = DISEASE_DATA[disease]
    if lang not in disease_info:
        lang = "en"

    return jsonify({
        "status": "success",
        "data": disease_info[lang]
    })

# --------------------
# Model loading (robust)
# --------------------
def try_load_model():
    global cnn_model
    loaded = False

    # Candidate: auto-detected file, relative to script, or parent folder
    candidate_paths = []

    if model_path_auto:
        candidate_paths.append(model_path_auto)
    candidate_paths += [
        ROOT / MODEL_FILENAME,
        ROOT.parent / MODEL_FILENAME
    ]

    # Filter unique and existing candidates
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
            print(f"Attempting to load model from: {p}")
            state = torch.load(str(p), map_location=device)
            # If the saved file is a state_dict (dict of params)
            if isinstance(state, dict) and any(k.startswith('module.') or k in CNN_ViT_Fusion(num_classes=NUM_CLASSES, pretrained=False, freeze_backbones=False).state_dict() for k in state.keys()):
                model_instance = CNN_ViT_Fusion(num_classes=NUM_CLASSES, pretrained=False, freeze_backbones=False).to(device)
                model_instance.load_state_dict(state)
                model_instance.eval()
                cnn_model = model_instance
                loaded = True
                print("Model loaded (state_dict).")
                break
            # Else the saved object could be a full model
            elif isinstance(state, nn.Module):
                cnn_model = state.to(device)
                cnn_model.eval()
                loaded = True
                print("Model loaded (full model object).")
                break
            else:
                # Try to set as state_dict anyway (best-effort)
                model_instance = CNN_ViT_Fusion(num_classes=NUM_CLASSES, pretrained=False, freeze_backbones=False).to(device)
                model_instance.load_state_dict(state)
                model_instance.eval()
                cnn_model = model_instance
                loaded = True
                print("Model loaded (fallback).")
                break
        except Exception as e:
            print(f"Failed loading from {p}: {e}")

    if not loaded:
        print("WARNING: Could not locate or load the model file. Ensure model_best.pth exists in project root or a subfolder. Searched candidates:", filtered)

# Attempt load at startup
try_load_model()

# --------------------
# Run the app
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

