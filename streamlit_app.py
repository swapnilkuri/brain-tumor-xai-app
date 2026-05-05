import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from models.model_factory import get_model
import gdown
import os

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

MODEL_NAME_MAP = {
    "DenseNet121": "densenet121",
    "ResNet50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2": "mobilenet_v2",
}

MODEL_PATHS = {
    "DenseNet121": ("https://drive.google.com/uc?id=12ASdxYOzN8IsHyAu2tfjETCsLFo-vNDz", "densenet.pth"),
    "ResNet50": ("https://drive.google.com/uc?id=1Enuecoe_TCrZJ3EUZVEGfwDohg8GqIMG", "resnet.pth"),
    "EfficientNet-B0": ("https://drive.google.com/uc?id=1iLPjoDgegFYLgxw6B3cD6Vnfd_iQdX07", "efficientnet.pth"),
    "MobileNetV2": ("https://drive.google.com/uc?id=1Em7OfSqZbpdjceVRtNG-Cn4kBvsdynT9", "mobilenet.pth"),
}

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# REMOVE INPLACE RELU
# ======================
def remove_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

# ======================
# DOWNLOAD MODEL
# ======================
def load_model_from_url(url, filename):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    return torch.load(filename, map_location=DEVICE, weights_only=False)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model(name):
    model_name = MODEL_NAME_MAP[name]
    model = get_model(model_name, 3, pretrained=False)

    url, file = MODEL_PATHS[name]
    state_dict = load_model_from_url(url, file)

    model.load_state_dict(state_dict)
    remove_inplace_relu(model)

    model.to(DEVICE)
    model.eval()
    return model

# ======================
# TARGET LAYER
# ======================
def get_target_layer(model, name):
    if "densenet" in name:
        return model.features[-1]
    elif "resnet" in name:
        return model.layer4[-1]
    elif "efficientnet" in name:
        return model.features[-1]
    elif "mobilenet" in name:
        return model.features[-1]
    return list(model.modules())[-1]

# ======================
# GRAD-CAM (FINAL FIX)
# ======================
def grad_cam(model, img_tensor, model_name):
    target_layer = get_target_layer(model, model_name)

    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    handle = target_layer.register_forward_hook(forward_hook)

    img_tensor = img_tensor.clone().detach().requires_grad_(True)

    output = model(img_tensor)
    class_idx = output.argmax()
    score = output[:, class_idx]

    grads = torch.autograd.grad(
        outputs=score,
        inputs=activations[0],
        grad_outputs=torch.ones_like(score),
        retain_graph=True
    )[0]

    acts = activations[0]

    grads = grads.detach().cpu().numpy()
    acts = acts.detach().cpu().numpy()

    weights = np.mean(grads, axis=(2, 3))
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * acts[0, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    handle.remove()

    return cam

# ======================
# UI
# ======================
st.set_page_config(layout="wide")
st.title("Brain Tumor Classification with Explainability & Robustness")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
mode = st.radio("Mode", ["Single Model", "Compare All Models"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input MRI", width=300)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    if mode == "Single Model":
        selected_model = st.selectbox("Select Model", list(MODEL_NAME_MAP.keys()))
        model = load_model(selected_model)

        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        pred = CLASS_NAMES[np.argmax(probs)]
        st.subheader(f"{selected_model} Prediction: {pred}")

        cam = grad_cam(model, img_tensor, MODEL_NAME_MAP[selected_model])

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM")

    else:
        cols = st.columns(4)

        for i, name in enumerate(MODEL_NAME_MAP.keys()):
            model = load_model(name)

            with torch.no_grad():
                out = model(img_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            pred = CLASS_NAMES[np.argmax(probs)]

            cam = grad_cam(model, img_tensor, MODEL_NAME_MAP[name])
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0)

            with cols[i]:
                st.markdown(f"### {name}")
                st.write(pred)
                st.image(overlay)
