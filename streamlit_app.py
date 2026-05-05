import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

MODEL_NAME_MAP = {
    "DenseNet121": "densenet121",
    "ResNet50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2": "mobilenet_v2"
}

MODEL_URLS = {
    "densenet121": "https://drive.google.com/uc?id=12ASdxYOzN8IsHyAu2tfjETCsLFo-vNDz",
    "resnet50": "https://drive.google.com/uc?id=1Enuecoe_TCrZJ3EUZVEGfwDohg8GqIMG",
    "efficientnet_b0": "https://drive.google.com/uc?id=1iLPjoDgegFYLgxw6B3cD6Vnfd_iQdX07",
    "mobilenet_v2": "https://drive.google.com/uc?id=1Em7OfSqZbpdjceVRtNG-Cn4kBvsdynT9"
}

os.makedirs("models", exist_ok=True)

# ---------------- DOWNLOAD ----------------
def download_model(name):
    path = f"models/{name}.pth"
    if not os.path.exists(path):
        gdown.download(MODEL_URLS[name], path, quiet=False)
    return path

# ---------------- MODEL ----------------
def get_model(name):
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 3)

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)

    return model

@st.cache_resource
def load_model(name):
    try:
        path = download_model(name)
        model = get_model(name)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    except:
        return None

# ---------------- PREPROCESS ----------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)

# ---------------- PREDICT ----------------
def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# ---------------- TARGET LAYER ----------------
def get_target_layer(model, name):
    if name == "densenet121":
        return model.features.denseblock4

    elif name == "resnet50":
        return model.layer4

    elif name == "efficientnet_b0":
        return model.features[-1][0]

    elif name == "mobilenet_v2":
        return model.features[-1]

# ---------------- GRAD CAM ----------------
def grad_cam(model, img_tensor, model_name):

    model.eval()
    target_layer = get_target_layer(model, model_name)

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(img_tensor)
        class_idx = output.argmax(dim=1)

        model.zero_grad()
        output[0, class_idx].backward()

        if activations is None or gradients is None:
            return None

        acts = activations.detach().cpu().numpy()[0]
        grads = gradients.detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * acts, axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    except:
        return None

    finally:
        handle_f.remove()
        handle_b.remove()

# ---------------- UI ----------------
st.title("Brain Tumor Classification with Explainability")

img_file = st.file_uploader("Upload MRI", type=["jpg", "png"])

mode = st.radio("Mode", ["Single Model", "Compare All Models"])
selected = st.selectbox("Select Model", list(MODEL_NAME_MAP.keys()))

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, width=250)

    tensor = preprocess(image)

    def run(ui_name, name):

        model = load_model(name)

        if model is None:
            st.error(f"Model failed: {name}")
            return

        probs = predict(model, tensor)
        pred = CLASS_NAMES[np.argmax(probs)]
        conf = np.max(probs)

        st.subheader(f"{ui_name}: {pred}")
        st.write(f"Confidence: {conf:.4f}")

        # -------- SMALL GRAPH --------
        fig, ax = plt.subplots(figsize=(3,1.8))
        ax.bar(CLASS_NAMES, probs)
        ax.set_ylim(0,1)
        ax.set_title("Confidence", fontsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # -------- GRAD CAM --------
        cam = grad_cam(model, tensor.clone(), name)

        if cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(
                np.array(image.resize((224,224))),
                0.7,
                heatmap,
                0.3,
                0
            )

            st.image(overlay, caption="Grad-CAM")
        else:
            st.warning("Grad-CAM failed")

    if mode == "Single Model":
        run(selected, MODEL_NAME_MAP[selected])
    else:
        cols = st.columns(4)
        for i,(ui,name) in enumerate(MODEL_NAME_MAP.items()):
            with cols[i]:
                run(ui, name)
