import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
import urllib.request 

st.set_page_config(page_title="AI Brain Tumor Segmentation", layout="wide")
st.title("🧠 Medical Computer Vision: AI Brain Tumor Segmentation Tool")
st.write("---") 

st.markdown("""
**Developer Note:** This deep learning dashboard utilizes a **U-Net architecture with a ResNet34 backbone** trained in PyTorch.
It performs pixel-level semantic segmentation to identify lower-grade glioma (LGG) tumor regions from Brain MRI scans.
""") 

@st.cache_resource
def download_model_weights():
weights_path = "brain_tumor_unet.pth"
file_url = "https://github.com/inshaak11/Brain-Tumor-Segmentation-AI/raw/main/brain_tumor_unet.pth"
if not os.path.exists(weights_path):
with st.spinner("Downloading AI Engine Model Weights (87MB)... Please wait."):
try:
urllib.request.urlretrieve(file_url, weights_path)
except Exception as e:
pass
return weights_path 

@st.cache_resource
def load_unet_model():
download_model_weights()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
encoder_name="resnet34",
encoder_weights=None,
in_channels=3,
classes=1
)
weights_file = "brain_tumor_unet.pth"
if os.path.exists(weights_file):
model.load_state_dict(torch.load(weights_file, map_location=device))
model.to(device)
model.eval()
return model, device 

try:
model, device = load_unet_model()
st.sidebar.success("✅ AI Engine running successfully!")
except Exception as e:
st.sidebar.error("⚠️ System initializing. Please upload or configure target weights.") 

st.sidebar.title("Clinical Information")
st.sidebar.info("""
**Architecture:** U-Net (ResNet34 Encoder)
**Target Disease:** Lower-Grade Glioma (LGG)
**Dataset Reference:** TCGA Lower Grade Glioma
""") 

st.sidebar.warning("""
⚠️ **Clinical Disclaimer:**
This software is developed strictly for educational, portfolio demonstration, and screening exploratory purposes. It is NOT an FDA-approved medical diagnostic tool and must never replace official professional evaluation by a healthcare provider.
""") 

uploaded_file = st.file_uploader("Upload a Brain MRI Scan Slice (.tif, .png, .jpg)", type=["tif", "png", "jpg"]) 

if uploaded_file is not None:
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB) 

col1, col2, col3 = st.columns(3)

with col1:
st.subheader("1. Input Brain MRI")
st.image(original_rgb, use_container_width=True)

with col2:
st.subheader("2. AI Segmentation Processing")
with st.spinner("AI analyzing tissue structures..."):
h, w, c = original_rgb.shape
resized = cv2.resize(original_rgb, (256, 256))
normalized = resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalized = (normalized - mean) / std
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).cpu().squeeze().numpy()
        predicted_binary_mask = (prob_mask > 0.5).astype(np.uint8)

    predicted_binary_mask = cv2.resize(predicted_binary_mask, (w, h))

    st.success("Analysis Complete!")
    st.image(predicted_binary_mask * 255, caption="Generated Tumor Mask", use_container_width=True)

with col3:
st.subheader("3. Clinical Overlay Result")
overlay = original_rgb.copy()
for channel_idx, color_val in enumerate([255, 0, 0]):
    overlay[:, :, channel_idx] = np.where(predicted_binary_mask == 1, color_val, overlay[:, :, channel_idx])

blended = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)
st.image(blended, caption="Red Highlight = Segmented Tumor Location", use_container_width=True)

st.write("---")
tumor_pixel_count = np.sum(predicted_binary_mask == 1)
if tumor_pixel_count > 0:
st.metric(label="Tumor Region Detected", value="POSITIVE", delta=f"{tumor_pixel_count} Pixels Outlined", delta_color="inverse")
else:
st.metric(label="Tumor Region Detected", value="NEGATIVE", delta="Normal Tissue Structure")







   





