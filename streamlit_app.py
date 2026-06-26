import os
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms
import gdown
import cv2
import numpy as np

from xai.gradcam import GradCAM
from models.model_factory import get_model
from xai.gradcam_plus_plus import GradCAMPlusPlus
from xai.scorecam import ScoreCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary"
]

def get_target_layer(model, model_name):

    if model_name == "convnext_tiny":
        return model.features[-2]

    elif model_name == "densenet121":
        return model.features[-1]

    elif model_name == "resnet50":
        return model.layer4[-1]

    else:
        raise ValueError(model_name)
    

    

MODEL_NAME_MAP = {
    "ConvNeXt-Tiny": "convnext_tiny",
    "DenseNet121": "densenet121",
    "ResNet50": "resnet50",
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
    )
}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


@st.cache_resource
def load_single_model(model_title):

    model_name = MODEL_NAME_MAP[model_title]

    model = get_model(
        model_name,
        num_classes=3,
        pretrained=False
    )

    url, filename = MODEL_PATHS[model_title]

    state_dict = load_model_from_url(url,filename)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.to(DEVICE)

    model.eval()

    return model


def load_model_from_url(url, filename):

    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

    return torch.load(
        filename,
        map_location=DEVICE,
        weights_only=False
    )

@st.cache_resource
def load_models():

    return {

        "convnext_tiny":
            load_single_model("ConvNeXt-Tiny"),

        "densenet121":
            load_single_model("DenseNet121"),

        "resnet50":
            load_single_model("ResNet50")
    }

def ensemble_predict(models, img_tensor):

    probs = []

    with torch.no_grad():

        for model in models.values():

            output = model(img_tensor)

            probs.append(
                F.softmax(output, dim=1)
            )

    avg_prob = torch.mean(
        torch.stack(probs),
        dim=0
    )

    confidence, prediction = torch.max(
        avg_prob,
        dim=1
    )

    return (
        prediction.item(),
        confidence.item()
    )

def ensemble_gradcampp(image, img_tensor, models):

    cams = []

    for model_name, model in models.items():

        target_layer = get_target_layer(
            model,
            model_name
        )

        gradcampp = GradCAMPlusPlus(
            model,
            target_layer
        )

        cam = gradcampp.generate(img_tensor)

        cams.append(cam)

    ensemble_cam = np.mean(cams, axis=0)

    return overlay_heatmap(
        image,
        ensemble_cam
    )

def ensemble_scorecam(image, img_tensor, models):

    cams = []

    for model_name, model in models.items():

        target_layer = get_target_layer(
            model,
            model_name
        )

        scorecam = ScoreCAM(
            model,
            target_layer
        )

        cam = scorecam.generate(img_tensor)

        cams.append(cam)

    ensemble_cam = np.mean(cams, axis=0)

    return overlay_heatmap(
        image,
        ensemble_cam
    )

def overlay_heatmap(image, cam):

    image = np.array(image.resize((224, 224)))

    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


def ensemble_gradcam(image, img_tensor, models):

    cams = []

    for model_name, model in models.items():

        target_layer = get_target_layer(
            model,
            model_name
        )

        gradcam = GradCAM(
            model,
            target_layer
        )

        cam = gradcam.generate(img_tensor)

        cams.append(cam)

    ensemble_cam = np.mean(cams, axis=0)

    return overlay_heatmap(
        image,
        ensemble_cam
    )

st.set_page_config(layout="wide")

st.title("Brain Tumor Classification using Deep Learning Soft Voting Ensemble")

uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        width=350
    )

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with st.spinner("Loading ensemble..."):

        models = load_models()
        
pred, conf = ensemble_predict(
    models,
    img_tensor
)

CONFIDENCE_THRESHOLD = 0.70

if conf < CONFIDENCE_THRESHOLD:

    st.error(
        "Unknown MRI image.\n\nConfidence is too low."
    )

    st.stop()
    st.success("Model : Soft Voting Ensemble")

    st.metric(
        "Prediction",
        CLASS_NAMES[pred]
    )

    st.metric(
        "Confidence",
        f"{conf*100:.2f}%"
    )

    st.subheader("Explainability")

gradcam_img = ensemble_gradcam(
    image,
    img_tensor,
    models
)

gradcampp_img = ensemble_gradcampp(
    image,
    img_tensor,
    models
)

scorecam_img = ensemble_scorecam(
    image,
    img_tensor,
    models
)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        gradcam_img,
        caption="Grad-CAM",
        use_container_width=True
    )

with col2:
    st.image(
        gradcampp_img,
        caption="Grad-CAM++",
        use_container_width=True
    )

with col3:
    st.image(
        scorecam_img,
        caption="Score-CAM",
        use_container_width=True
    )
