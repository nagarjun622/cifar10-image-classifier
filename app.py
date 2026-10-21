
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ CIFAR-10 Image Classification")
st.write(
    "Upload an image to classify it into one of the "
    "10 CIFAR-10 categories."
)


# --------------------------------------------------
# 2. Model architectures
# These must match the notebook architectures.
# --------------------------------------------------

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class ImprovedCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(model_name, num_classes):
    if model_name == "Baseline CNN":
        return CIFAR10CNN(num_classes)

    elif model_name == "Improved CNN":
        return ImprovedCIFAR10CNN(num_classes)

    elif model_name == "Pretrained ResNet-18":
        model = resnet18(weights=None)

        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        model.maxpool = nn.Identity()

        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )

        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------
# 3. Load the saved model
# --------------------------------------------------

MODEL_PATH = "models/deployment_model.pth"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}. "
            "Place deployment_model.pth inside the models folder."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["classes"]
    normalization = checkpoint["normalization"]

    model = create_model(model_name, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_name, class_names, normalization


try:
    model, model_name, classes, normalization = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


st.caption(f"Model in use: {model_name}")


# --------------------------------------------------
# 4. Image preprocessing
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=normalization["mean"],
        std=normalization["std"]
    )
])


# --------------------------------------------------
# 5. Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        input_tensor = preprocess(image).unsqueeze(0)

        # ------------------------------------------
        # 6. Make prediction
        # ------------------------------------------

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = probabilities.argmax().item()
        predicted_class = classes[predicted_index]
        confidence = probabilities[predicted_index].item()

        with col2:
            st.subheader("Prediction")
            st.success(f"Predicted class: **{predicted_class}**")
            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

        # ------------------------------------------
        # 7. Top-5 predictions
        # ------------------------------------------

        st.subheader("Top Predictions")

        top_count = min(5, len(classes))
        top_probs, top_indices = torch.topk(
            probabilities, top_count
        )

        prediction_df = pd.DataFrame({
            "Class": [
                classes[index] for index in top_indices.tolist()
            ],
            "Confidence (%)": [
                value * 100 for value in top_probs.tolist()
            ]
        })

        st.bar_chart(
            prediction_df.set_index("Class")["Confidence (%)"]
        )

        st.dataframe(
            prediction_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "CIFAR-10 contains small 32×32 images. "
            "Images outside this dataset's categories or style "
            "may be classified incorrectly. Confidence is not "
            "a guarantee that a prediction is correct."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
