import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import gdown
import matplotlib.pyplot as plt

from models.model_factory import get_model

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

MODEL_URLS = {
    "DenseNet121": "https://drive.google.com/uc?id=12ASdxYOzN8IsHyAu2tfjETCsLFo-vNDz",
    "ResNet50": "https://drive.google.com/uc?id=1Enuecoe_TCrZJ3EUZVEGfwDohg8GqIMG",
    "EfficientNet-B0": "https://drive.google.com/uc?id=1iLPjoDgegFYLgxw6B3cD6Vnfd_iQdX07",
    "MobileNetV2": "https://drive.google.com/uc?id=1Em7OfSqZbpdjceVRtNG-Cn4kBvsdynT9"
}

MODEL_FILES = {
    name: f"{name}.pth" for name in MODEL_URLS
}

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# DOWNLOAD MODEL
# =====================
def download_model(name):
    path = MODEL_FILES[name]
    if not os.path.exists(path):
        gdown.download(MODEL_URLS[name], path, quiet=False)
    return path

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model(name):
    path = download_model(name)

    model_name = name.lower().replace("-", "").replace(" ", "")
    model = get_model(model_name, 3, pretrained=False)

    state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

# =====================
# PREDICTION
# =====================
def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# =====================
# GET LAST CONV LAYER
# =====================
def get_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module

# =====================
# GRAD-CAM
# =====================
def grad_cam(model, img_tensor):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.clone())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].clone())

    target_layer = get_target_layer(model)

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam

# =====================
# GRAD-CAM++
# =====================
def grad_cam_pp(model, img_tensor):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.clone())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].clone())

    target_layer = get_target_layer(model)

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    grads_power = grads ** 2
    weights = grads_power / (2 * grads_power + acts * grads ** 3 + 1e-8)
    weights = weights.mean(dim=(2, 3), keepdim=True)

    cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam

# =====================
# HEATMAP OVERLAY
# =====================
def overlay(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# =====================
# PLOT
# =====================
def plot_probs(probs):
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs)
    ax.set_title("Confidence Distribution")
    st.pyplot(fig)

# =====================
# UI
# =====================
st.title("Brain Tumor Classification with Explainability")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

mode = st.radio("Mode", ["Single Model", "Compare All Models"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input MRI")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_np = np.array(img)

    if mode == "Single Model":
        model_name = st.selectbox("Select Model", list(MODEL_URLS.keys()))
        model = load_model(model_name)

        probs = predict(model, img_tensor)
        pred = CLASS_NAMES[np.argmax(probs)]

        st.subheader(f"{model_name} Prediction: {pred}")
        st.write(f"Confidence: {np.max(probs):.4f}")

        plot_probs(probs)

        cam = grad_cam(model, img_tensor)
        cam_pp = grad_cam_pp(model, img_tensor)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Grad-CAM")
            st.image(overlay(img_np, cam))

        with col2:
            st.caption("Grad-CAM++")
            st.image(overlay(img_np, cam_pp))

    else:
        cols = st.columns(4)

        for i, name in enumerate(MODEL_URLS.keys()):
            model = load_model(name)
            probs = predict(model, img_tensor)
            pred = CLASS_NAMES[np.argmax(probs)]

            cam = grad_cam(model, img_tensor)

            with cols[i]:
                st.markdown(f"### {name}")
                st.write(pred)
                st.write(f"{np.max(probs):.3f}")
                st.image(overlay(img_np, cam))
