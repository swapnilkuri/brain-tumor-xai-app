import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary']
DEVICE = torch.device("cpu")

# ================= MODEL FACTORY =================
def get_model(name):
    if name == "DenseNet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 3)

    elif name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)

    elif name == "EfficientNet-B0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    elif name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# ================= LOAD MODEL =================
@st.cache_resource
def load_model(name):
    model = get_model(name)

    path_map = {
        "DenseNet121": "models/densenet121.pth",
        "ResNet50": "models/resnet50.pth",
        "EfficientNet-B0": "models/efficientnetb0.pth",
        "MobileNetV2": "models/mobilenetv2.pth"
    }

    path = path_map[name]

    if not os.path.exists(path):
        st.error(f"Model file missing: {path}")
        return None

    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ================= PREPROCESS =================
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


# ================= GRAD-CAM =================
def grad_cam(model, img_tensor):

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

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


# ================= UI =================
st.title("Brain Tumor Classification with Explainability")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

mode = st.radio("Mode", ["Single Model", "Compare All Models"])

MODEL_LIST = ["DenseNet121", "ResNet50", "EfficientNet-B0", "MobileNetV2"]

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input MRI", width=200)

    img_tensor = preprocess(image)

    def run_model(model_name):
        model = load_model(model_name)
        if model is None:
            return

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = CLASS_NAMES[np.argmax(probs)]
        confidence = np.max(probs)

        st.subheader(f"{model_name} Prediction: {pred_class}")
        st.write(f"Confidence: {confidence:.4f}")

        # ===== FIXED SMALL GRAPH =====
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(CLASS_NAMES, probs)
        ax.set_ylim(0, 1)
        ax.set_title("Confidence")
        st.pyplot(fig)
        plt.close(fig)

        # ===== GRAD-CAM =====
        cam = grad_cam(model, img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            np.array(image.resize((224, 224))),
            0.7,
            heatmap,
            0.3,
            0
        )

        st.image(overlay, caption="Grad-CAM")

    if mode == "Single Model":
        selected_model = st.selectbox("Select Model", MODEL_LIST)
        run_model(selected_model)

    else:
        cols = st.columns(4)
        for i, m in enumerate(MODEL_LIST):
            with cols[i]:
                run_model(m)
