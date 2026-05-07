import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from models.model_factory import get_model
from xai.gradcam import GradCAM
from xai.gradcam_plus_plus import GradCAMPlusPlus
from xai.scorecam import ScoreCAM

import gdown
import os
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

MODEL_NAME_MAP = {
    "ConvNeXt-Tiny": "convnext_tiny",
    "DenseNet121": "densenet121",
    "ResNet50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2": "mobilenet_v2",
}

MODEL_PATHS = {
    "ConvNeXt-Tiny": (
        "https://drive.google.com/uc?id=1H88X-CmdtVycmb1IzgNsy9kGG2Nt9WKj",
        "convnext_tiny.pth"
    ),

    "DenseNet121": (
        "https://drive.google.com/uc?id=12ASdxYOzN8IsHyAu2tfjETCsLFo-vNDz",
        "densenet.pth"
    ),

    "ResNet50": (
        "https://drive.google.com/uc?id=1Enuecoe_TCrZJ3EUZVEGfwDohg8GqIMG",
        "resnet.pth"
    ),

    "EfficientNet-B0": (
        "https://drive.google.com/uc?id=1iLPjoDgegFYLgxw6B3cD6Vnfd_iQdX07",
        "efficientnet.pth"
    ),

    "MobileNetV2": (
        "https://drive.google.com/uc?id=1Em7OfSqZbpdjceVRtNG-Cn4kBvsdynT9",
        "mobilenet.pth"
    ),
}

# ======================
# IMAGE TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ======================
# REMOVE INPLACE RELU
# ======================
def remove_inplace_relu(model):

    for module in model.modules():

        if isinstance(module, nn.ReLU):

            module.inplace = False

# ======================
# DOWNLOAD MODEL
# ======================
def load_model_from_url(url, filename):

    if not os.path.exists(filename):

        gdown.download(url, filename, quiet=False)

    return torch.load(
        filename,
        map_location=DEVICE,
        weights_only=False
    )

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model(name):

    try:

        model_name = MODEL_NAME_MAP[name]

        model = get_model(
            model_name,
            3,
            pretrained=False
        )

        url, file = MODEL_PATHS[name]

        # Download if missing
        if not os.path.exists(file):

            gdown.download(
                url,
                file,
                quiet=False
            )

        # Load checkpoint safely
        checkpoint = torch.load(
            file,
            map_location=DEVICE
        )

        # Handle state_dict wrapper
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(
            checkpoint,
            strict=False
        )

        remove_inplace_relu(model)

        model.to(DEVICE)

        model.eval()

        return model

    except Exception as e:

        st.error(f"Failed loading {name}: {e}")

        return None

# ======================
# TARGET LAYER
# ======================
def get_target_layer(model, name):

    if name == "convnext_tiny":

        return model.features[-1]

    elif name == "densenet121":

        return model.features[-1]

    elif name == "resnet50":

        return model.layer4[-1]

    elif name == "efficientnet_b0":

        return model.features[-1]

    elif name == "mobilenet_v2":

        return model.features[-1]

# ======================
# CONFIDENCE GRAPH
# ======================
def show_confidence_graph(probs):

    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.bar(CLASS_NAMES, probs)

    ax.set_ylim(0, 1)

    ax.set_title("Confidence", fontsize=10)

    ax.tick_params(axis='x', labelsize=8)

    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)

# ======================
# GENERATE GRADCAM
# ======================
def generate_gradcam(model, img_tensor, model_name):

    target_layer = get_target_layer(
        model,
        model_name
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(img_tensor)

    return cam

# ======================
# GENERATE GRADCAM++
# ======================
def generate_gradcam_pp(model, img_tensor, model_name):

    target_layer = get_target_layer(
        model,
        model_name
    )

    gradcam_pp = GradCAMPlusPlus(
        model,
        target_layer
    )

    cam = gradcam_pp.generate(img_tensor)

    return cam

# ======================
# GENERATE SCORECAM
# ======================
def generate_scorecam(model, img_tensor, model_name):

    target_layer = get_target_layer(
        model,
        model_name
    )

    scorecam = ScoreCAM(
        model,
        target_layer
    )

    cam = scorecam.generate(img_tensor)

    return cam

# ======================
# OVERLAY HEATMAP
# ======================
def overlay_heatmap(image, cam):

    cam = cv2.resize(cam, (224, 224))

    cam = cam - np.min(cam)

    cam = cam / (np.max(cam) + 1e-8)

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    image_np = np.array(
        image.resize((224, 224))
    )

    image_np = image_np.astype(np.uint8)

    heatmap = heatmap.astype(np.uint8)

    # IMPORTANT FIX
    if len(image_np.shape) == 2:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )

    if image_np.shape[2] == 4:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGBA2RGB
        )

    overlay = cv2.addWeighted(
        image_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay

# ======================
# UI
# ======================
st.set_page_config(layout="wide")

st.title(
    "Brain Tumor Classification with Explainability & Robustness"
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

mode = st.radio(
    "Mode",
    ["Single Model", "Compare All Models"]
)

# ======================
# MAIN
# ======================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Input MRI",
        width=180
    )

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ==================================================
    # SINGLE MODEL
    # ==================================================
    if mode == "Single Model":

        selected_model = st.selectbox(
            "Select Model",
            list(MODEL_NAME_MAP.keys())
        )

        model = load_model(selected_model)

        if model is None:
            st.stop()

        with torch.no_grad():

            out = model(img_tensor)

            probs = torch.softmax(
                out,
                dim=1
            ).cpu().numpy()[0]

        pred = CLASS_NAMES[np.argmax(probs)]

        st.subheader(f"{selected_model}: {pred}")

        st.write(
            f"Confidence: {np.max(probs):.4f}"
        )

        show_confidence_graph(probs)

        try:

            cam1 = generate_gradcam(
                model,
                img_tensor,
                MODEL_NAME_MAP[selected_model]
            )

            overlay1 = overlay_heatmap(image, cam1)

            st.image(
                overlay1,
                caption="Grad-CAM"
            )

            cam2 = generate_gradcam_pp(
                model,
                img_tensor,
                MODEL_NAME_MAP[selected_model]
            )

            overlay2 = overlay_heatmap(image, cam2)

            st.image(
                overlay2,
                caption="Grad-CAM++"
            )

            cam3 = generate_scorecam(
                model,
                img_tensor,
                MODEL_NAME_MAP[selected_model]
            )

            overlay3 = overlay_heatmap(image, cam3)

            st.image(
                overlay3,
                caption="Score-CAM"
            )

        except Exception as e:

            st.warning(f"XAI failed: {e}")

       # ==================================================
    # COMPARE ALL MODELS
    # ==================================================
    else:

        model_names = list(MODEL_NAME_MAP.keys())

        cols = st.columns(len(model_names))

        for idx, name in enumerate(model_names):

            model = load_model(name)

            if model is None:
                continue

            with torch.no_grad():

                out = model(img_tensor)

                probs = torch.softmax(
                    out,
                    dim=1
                ).cpu().numpy()[0]

            pred = CLASS_NAMES[np.argmax(probs)]

            with cols[idx]:

                st.markdown(f"## {name}")

                st.write(pred)

                st.write(
                    f"Confidence: {np.max(probs):.4f}"
                )

                show_confidence_graph(probs)

                try:

                    # ======================
                    # GRAD-CAM
                    # ======================
                    cam1 = generate_gradcam(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[name]
                    )

                    overlay1 = overlay_heatmap(
                        image,
                        cam1
                    )

                    st.image(
                        overlay1,
                        caption="Grad-CAM"
                    )

                    # ======================
                    # GRAD-CAM++
                    # ======================
                    cam2 = generate_gradcam_pp(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[name]
                    )

                    overlay2 = overlay_heatmap(
                        image,
                        cam2
                    )

                    st.image(
                        overlay2,
                        caption="Grad-CAM++"
                    )

                    # ======================
                    # SCORE-CAM
                    # ======================
                    cam3 = generate_scorecam(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[name]
                    )

                    overlay3 = overlay_heatmap(
                        image,
                        cam3
                    )

                    st.image(
                        overlay3,
                        caption="Score-CAM"
                    )

                except Exception as e:

                    st.warning(
                        f"XAI failed: {e}"
                    )
