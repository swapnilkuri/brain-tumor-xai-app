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

    model_name = MODEL_NAME_MAP[name]

    model = get_model(
        model_name,
        3,
        pretrained=False
    )

    url, file = MODEL_PATHS[name]

    state_dict = load_model_from_url(url, file)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)

    remove_inplace_relu(model)

    model.to(DEVICE)

    model.eval()

    return model

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
# MAIN APP
# ======================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    st.sidebar.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # ======================
    # SINGLE MODEL
    # ======================
    if mode == "Single Model":

        selected_model_name = st.selectbox(
            "Select Model",
            list(MODEL_NAME_MAP.keys())
        )

        model = load_model(selected_model_name)

        with torch.no_grad():

            out = model(img_tensor)

            probs = torch.softmax(out, dim=1)

            conf, pred = torch.max(probs, 1)

            confidence = conf.item()

            pred_idx = pred.item()

        THRESHOLD = 0.70

        if confidence < THRESHOLD:

            st.error("Unknown / Invalid MRI Image")

            st.warning(
                "The uploaded image does not match the trained tumor classes."
            )

        else:

            st.header(
                f"Result: {CLASS_NAMES[pred_idx]}"
            )

            st.write(
                f"Confidence: **{confidence:.2%}**"
            )

            col1, col2, col3 = st.columns(3)

            try:

                with col1:

                    cam1 = generate_gradcam(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[selected_model_name]
                    )

                    st.image(
                        overlay_heatmap(image, cam1),
                        caption="Grad-CAM"
                    )

                with col2:

                    cam2 = generate_gradcam_pp(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[selected_model_name]
                    )

                    st.image(
                        overlay_heatmap(image, cam2),
                        caption="Grad-CAM++"
                    )

                with col3:

                    cam3 = generate_scorecam(
                        model,
                        img_tensor,
                        MODEL_NAME_MAP[selected_model_name]
                    )

                    st.image(
                        overlay_heatmap(image, cam3),
                        caption="Score-CAM"
                    )

            except Exception as e:

                st.error(
                    f"XAI Generation Error: {e}"
                )

    # ======================
    # COMPARE ALL MODELS
    # ======================
    else:

        st.info(
            "Comparing all models. This may take a moment..."
        )

        THRESHOLD = 0.70

        for name in MODEL_NAME_MAP.keys():

            st.divider()

            st.subheader(f"Model: {name}")

            model = load_model(name)

            with torch.no_grad():

                out = model(img_tensor)

                probs = torch.softmax(out, dim=1)

                conf, pred = torch.max(probs, 1)

                confidence = conf.item()

                pred_idx = pred.item()

            res_col, cam_col1, cam_col2, cam_col3 = st.columns(
                [1, 2, 2, 2]
            )

            with res_col:

                if confidence < THRESHOLD:

                    st.error("Invalid MRI")

                    st.write(
                        f"Low Confidence: {confidence:.2%}"
                    )

                else:

                    st.metric(
                        "Prediction",
                        CLASS_NAMES[pred_idx]
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

            if confidence >= THRESHOLD:

                try:

                    with cam_col1:

                        c1 = generate_gradcam(
                            model,
                            img_tensor,
                            MODEL_NAME_MAP[name]
                        )

                        st.image(
                            overlay_heatmap(image, c1),
                            caption="Grad-CAM"
                        )

                    with cam_col2:

                        c2 = generate_gradcam_pp(
                            model,
                            img_tensor,
                            MODEL_NAME_MAP[name]
                        )

                        st.image(
                            overlay_heatmap(image, c2),
                            caption="Grad-CAM++"
                        )

                    with cam_col3:

                        c3 = generate_scorecam(
                            model,
                            img_tensor,
                            MODEL_NAME_MAP[name]
                        )

                        st.image(
                            overlay_heatmap(image, c3),
                            caption="Score-CAM"
                        )

                except Exception as e:

                    st.warning(
                        f"Interpretability failed for {name}: {e}"
                    )

            del model

            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()

else:

    st.write(
        "Please upload an MRI image (JPG/PNG) to begin."
    )
