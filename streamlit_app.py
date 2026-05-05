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
# IMAGE PREPROCESS
# -------------------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# -------------------------
# PREDICTION
# -------------------------
def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).numpy()[0]
    return probs

# -------------------------
# GRAD-CAM (SAFE VERSION)
# -------------------------
def grad_cam(model, img_tensor):

    img_tensor.requires_grad = True

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    gradients = img_tensor.grad[0].cpu().numpy()
    cam = np.mean(gradients, axis=0)

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

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
    st.image(image, caption="Input MRI", width=300)

    tensor = preprocess(image)

    if mode == "Single Model":
        name = MODEL_NAME_MAP[selected_model]
        model = load_model(name)

        probs = predict(model, tensor)
        pred_class = CLASS_NAMES[np.argmax(probs)]

        st.subheader(f"{selected_model} Prediction: {pred_class}")
        st.write(f"Confidence: {np.max(probs):.4f}")

        # chart
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs)
        ax.set_title("Confidence Distribution")
        st.pyplot(fig)

        # grad cam
        cam = grad_cam(model, tensor.clone())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM")

    else:
        cols = st.columns(4)

        for i, (ui_name, name) in enumerate(MODEL_NAME_MAP.items()):
            model = load_model(name)
            probs = predict(model, tensor)

            pred_class = CLASS_NAMES[np.argmax(probs)]
            cam = grad_cam(model, tensor.clone())

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0)

            with cols[i]:
                st.markdown(f"### {ui_name}")
                st.write(pred_class)
                st.write(f"{np.max(probs):.3f}")
                st.image(overlay)
