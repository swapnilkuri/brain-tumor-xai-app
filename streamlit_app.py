import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["glioma", "meningioma", "pituitary"]

MODEL_PATHS = {
    "densenet121": "models/densenet121.pth",
    "resnet50": "models/resnet50.pth",
    "efficientnet_b0": "models/efficientnetb0.pth",
    "mobilenet_v2": "models/mobilenetv2.pth"
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(name):
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 3)

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    else:
        raise ValueError(f"Unknown model: {name}")

    path = MODEL_PATHS[name]
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

# ---------------- TARGET LAYER ----------------
def get_target_layer(model, name):
    if name == "densenet121":
        return model.features[-1]

    elif name == "resnet50":
        return model.layer4[-1]

    elif name == "efficientnet_b0":
        return model.features[-1]

    elif name == "mobilenet_v2":
        return model.features[-1]

# ---------------- GRAD CAM ----------------
def grad_cam(model, x, model_name):
    target = get_target_layer(model, model_name)

    activations = []
    gradients = []

    def f_hook(m, i, o):
        activations.append(o)

    def b_hook(m, gi, go):
        gradients.append(go[0])

    h1 = target.register_forward_hook(f_hook)
    h2 = target.register_backward_hook(b_hook)

    out = model(x)
    cls = out.argmax(dim=1)

    model.zero_grad()
    out[0, cls].backward()

    acts = activations[0].detach().cpu().numpy()[0]
    grads = gradients[0].detach().cpu().numpy()[0]

    weights = grads.mean(axis=(1, 2))
    cam = (weights[:, None, None] * acts).sum(axis=0)

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam

# ---------------- PLOT FIXED SMALL GRAPH ----------------
def plot_confidence(probs):
    fig, ax = plt.subplots(figsize=(3, 2))  # ✅ FIXED SIZE

    ax.bar(CLASSES, probs)
    ax.set_ylim(0, 1)
    ax.set_title("Confidence", fontsize=10)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------------- RUN MODEL ----------------
def run_model(model_name, image):
    try:
        model = load_model(model_name)
    except:
        st.error(f"Model missing: {model_name}")
        return

    tensor = preprocess(image)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred = CLASSES[np.argmax(probs)]
    conf = float(np.max(probs))

    st.subheader(f"{model_name.upper()} Prediction: {pred}")
    st.write(f"Confidence: {conf:.4f}")

    plot_confidence(probs)

    # ---- Grad CAM ----
    cam = grad_cam(model, tensor.clone(), model_name)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM", use_container_width=False)

# ---------------- UI ----------------
st.title("Brain Tumor Classification with Explainability")

uploaded = st.file_uploader("Upload MRI Image")

mode = st.radio("Mode", ["Single Model", "Compare All Models"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input MRI", width=200)

    if mode == "Single Model":
        selected = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
        run_model(selected, img)

    else:
        cols = st.columns(2)

        for i, name in enumerate(MODEL_PATHS.keys()):
            with cols[i % 2]:
                run_model(name, img)
