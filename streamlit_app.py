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

# -------------------------
# CONFIG
# -------------------------
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

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model(name):
    path = f"models/{name}.pth"
    if not os.path.exists(path):
        with st.spinner(f"Downloading {name}..."):
            gdown.download(MODEL_URLS[name], path, quiet=False)
    return path

# -------------------------
# MODEL FACTORY
# -------------------------
def get_model(name):
    if name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 3)

    elif name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model(name):
    path = download_model(name)
    model = get_model(name)

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

# -------------------------
# PREPROCESS (FIXED)
# -------------------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# -------------------------
# PREDICTION
# -------------------------
def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# -------------------------
# REAL GRAD-CAM
# -------------------------
def grad_cam(model, img_tensor):

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # last conv layer
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0][0].cpu().numpy()
    acts = activations[0][0].cpu().numpy()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam

# -------------------------
# UI
# -------------------------
st.title("Brain Tumor Classification with Explainability")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

mode = st.radio("Mode", ["Single Model", "Compare All Models"])

selected_model = st.selectbox("Select Model", list(MODEL_NAME_MAP.keys()))

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input MRI", width=250)

    tensor = preprocess(image)

    def run_model(ui_name, name):
        model = load_model(name)
        probs = predict(model, tensor)

        pred_class = CLASS_NAMES[np.argmax(probs)]
        confidence = np.max(probs)

        st.subheader(f"{ui_name} Prediction: {pred_class}")
        st.write(f"Confidence: {confidence:.4f}")

        # SMALL GRAPH (FIXED)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(CLASS_NAMES, probs)
        ax.set_ylim(0,1)
        ax.set_title("Confidence")
        st.pyplot(fig)
        plt.close(fig)

        # Grad-CAM
        cam = grad_cam(model, tensor.clone())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            np.array(image.resize((224,224))),
            0.7,
            heatmap,
            0.3,
            0
        )

        st.image(overlay, caption="Grad-CAM")

    if mode == "Single Model":
        run_model(selected_model, MODEL_NAME_MAP[selected_model])

    else:
        cols = st.columns(4)

        for i, (ui_name, name) in enumerate(MODEL_NAME_MAP.items()):
            with cols[i]:
                run_model(ui_name, name)
