import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from torchsummary import summary
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import joblib
import base64
# from streamlit_option_menu import option_menu
from streamlit.components.v1 import html
# import hydralit_components as hc

@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_png_as_page_bg("MRI_Machine.jpg")

# st.markdown("""
#     <style>
#     .title {
#         background-color: none; /* Choose a contrasting background color */
#         color: black; /* Text color for the title */
#         padding: 5px; /* Padding around the title text */
#         border-radius: 50px; /* Rounded corners */
#         text-align: center; /* Center the title text */
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # HTML to display the title with the styled class
# st.markdown('<h1 class="title">Brain Tumor Classification Machine</h1>', unsafe_allow_html=True)

## Function to resize image to 224x224
def resize_image(image):
    return cv2.resize(image, (224, 224))

# Function to apply CLAHE with adjustable parameters
def apply_clahe(image, clip_limit, tile_grid_size):
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # If single-channel RGB, convert to 3-channel RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))  # Merge enhanced L-channel with A and B channels
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  # Convert back to RGB

# Function to apply Gaussian filter with adjustable parameters
def apply_gaussian_filter(image, filter_size, sigma):
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.GaussianBlur(image, (filter_size, filter_size), sigma)

# Function to calculate MSE
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Function to calculate RMSE
def calculate_rmse(image1, image2):
    mse = calculate_mse(image1, image2)
    return np.sqrt(mse)

# Function to calculate PSNR
def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Function to reset all session state variables related to images
def reset_image_states():
    st.session_state.uploaded_image = None
    st.session_state.resized_image = None
    st.session_state.clahe_image = None
    st.session_state.gaussian_image = None

# Initialize session state variables
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'resized_image' not in st.session_state:
    st.session_state.resized_image = None
if 'clahe_image' not in st.session_state:
    st.session_state.clahe_image = None
if 'gaussian_image' not in st.session_state:
    st.session_state.gaussian_image = None

# Load the VGG16 feature extractor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg_model = models.vgg16()
# vgg_model.classifier[-1] = nn.Identity()
# vgg_model.load_state_dict(torch.load('vgg16_feature_extractor.pth', map_location=device))
# vgg_model = vgg_model.to(device)
# vgg_model.eval()

# Load the PCA model
# pca = joblib.load('pca_model.pkl')

# Load the trained SVM classifier
# svm_classifier = joblib.load('svm_classifier.pkl')

# Define data transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


### Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid black;
    }
</style>
    """,
    unsafe_allow_html=True
)
selected_option = st.sidebar.radio(
    "Navigation",
    ["Home", "Preprocessing", "Classification", "About"],
    index=0
)


if selected_option == "Home":
#     st.markdown(
#     """
#     <style>
#     .floating-menu {
#         position: fixed;
#         top: 0px;
#         right: 100%;
#         transform: translateX(-50%);
#         z-index: 1000;
#         background-color: #808080;
#         border-radius: px;
#         box-shadow: 0 0px 0px rgba(0, 0, 0, 0);
#     }
#     .nav-link {
#         padding: 8px 0px;
#         background-color: #808080;
#         color: none;
#         border-radius: 0px;
#         border: 0px solid white;
#         margin: 0 0px;
#         text-align: center;
#         font-weight: bold;
#         text-decoration: none;
#     }
#     .nav-link:hover {
#         background-color: #666666 !important;
#     }
#     .nav-link-selected {
#         background-color: #444444 !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


#     header = st.container()
#     header.title("Brain Tumor Classification Machine")
#     header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    st.markdown("""
        <style>
        .header-with-background {
            background-color: #808080; /* Background color for the header */
            color: white; /* Text color for the header */
            padding: 10px; /* Padding around the header */
            border-radius: 5px; /* Rounded corners */
            font-size: 25px; /* Font size for the header */
            font-weight: bold; /* Bold font */
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<p class='header-with-background'>KLASIFIKASI JENIS TUMOR OTAK MENINGIOMA, GLIOMA, DAN PITUITARI HYBRID VGG-16 DAN SVM UNTUK DIAGNOSIS PRAOPERASI</p>", unsafe_allow_html=True)
    st.markdown("<p><strong><span style='font-size:25px;'>Ariq Dreiki Hajjanto - 5023201057</span></strong></p>", unsafe_allow_html=True)
    st.markdown("<p><strong><span style='font-size: 20px;'>Dosen Pembimbing 1: Prof. Dr. Tri Arief Sardjono, S.T., M.T.</p>", unsafe_allow_html=True)
    st.markdown("<p><strong><span style='font-size: 20px;'>Dosen Pembimbing 2: Nada Fitrieyatul Hikmah, S.T., M.T.</p>", unsafe_allow_html=True)

elif selected_option == "Preprocessing":
    st.markdown("""
        <style>
        .section {
            background-color: #000000; /* Background color for the section */
            padding: 0px; /* Padding around the section */
            border-radius: 5px; /* Rounded corners */
        }
        </style>
        """, unsafe_allow_html=True)
        # Adding bullet points
    st.markdown("""
        <div class='section'>
            <p>&#x2757 Requirements for image preprocessing &#x2757</p>
            <ul>
                <li>Ensure the utilization of MRI images as the input for optimal results &#128444;</li>
                <li>Utilize greyscale images for accurate analysis and feature extraction &#x1F50D</li>
                <li>Limit the process to brain tumor images such as meningioma, glioma, and pituitary for focused research &#129504</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Always show the buttons
        st.sidebar.markdown("### Resize")
        resize_button = st.sidebar.button(':one: Resize to 224x224')

        # Default values for CLAHE parameters
        default_clip_limit = 1.0
        default_tile_grid_size = 8

        # Default values for Gaussian filter parameters
        default_sigma = 1.0
        default_filter_size = 3

        # CLAHE parameters
        st.sidebar.markdown("### CLAHE Parameters")
        clip_limit = st.sidebar.slider("Clip Limit", 0.0, 5.0, default_clip_limit)
        tile_grid_size = st.sidebar.selectbox("Tile Grid Size", [8, 16], index=0 if default_tile_grid_size == 8 else 1)
        clahe_button = st.sidebar.button(':two: Apply CLAHE')

        # Gaussian filter parameters
        st.sidebar.markdown("### Gaussian Filter Parameters")
        sigma = st.sidebar.slider("Sigma", 1.0, 3.0, default_sigma)
        filter_size = st.sidebar.selectbox("Filter Size", [3, 5, 7], index=0 if default_filter_size == 3 else 1)
        gaussian_button = st.sidebar.button(':three: Apply Gaussian Filter')

        # Add "Start All" button
        start_all_button = st.sidebar.button(':four: Start All')

        st.sidebar.markdown("### Reset")
        reset_button = st.sidebar.button("Reset Images")
        if reset_button:
            reset_image_states()

        # Step 1: Resize image to 224x224
        if resize_button or start_all_button:
            st.session_state.resized_image = resize_image(image_np)
            #st.subheader("Resize to 224x224")
            col1, col2 = st.columns(2)

        # Step 2: Apply CLAHE
        if (st.session_state.resized_image is not None and clahe_button) or start_all_button:
            st.session_state.clahe_image = apply_clahe(st.session_state.resized_image, clip_limit, tile_grid_size)
            #st.subheader("Apply CLAHE")
            col1, col2 = st.columns(2)

        # Step 3: Apply Gaussian filter
        if (st.session_state.clahe_image is not None and gaussian_button) or start_all_button:
            st.session_state.gaussian_image = apply_gaussian_filter(st.session_state.clahe_image, filter_size, sigma)
            #st.subheader("Apply Gaussian Filter")
            col1, col2 = st.columns(2)

        st.subheader("All Processed Images")
        col1, col2 = st.columns(2)

        # Original Image
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        # Resized Image (Step 1)
        with col2:
            if st.session_state.resized_image is not None:
                st.subheader("Resized Image (224x224)")
                st.image(st.session_state.resized_image, use_column_width=True)

        # After CLAHE (Step 2)
        with col1:
            if st.session_state.clahe_image is not None:
                st.subheader("After CLAHE")
                st.image(st.session_state.clahe_image, use_column_width=True)

        # After Gaussian Filter (Step 3)
        with col2:
            if st.session_state.gaussian_image is not None:
                st.subheader("After Gaussian Filter")
                st.image(st.session_state.gaussian_image, use_column_width=True)
                mse = calculate_mse(st.session_state.clahe_image, st.session_state.gaussian_image)
                rmse = calculate_rmse(st.session_state.clahe_image, st.session_state.gaussian_image)
                psnr = calculate_psnr(st.session_state.clahe_image, st.session_state.gaussian_image)

                st.subheader("Metrics After Gaussian Filter")
                st.write(f"MSE: {mse:.2f}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"PSNR: {psnr:.2f} dB")

            # Option to download the preprocessed image as JPEG
            if st.session_state.gaussian_image is not None:
                result_image = Image.fromarray(cv2.cvtColor(st.session_state.gaussian_image, cv2.COLOR_BGR2RGB))
                img_bytes = io.BytesIO()
                result_image.save(img_bytes, format='JPEG', quality=95)
                img_bytes.seek(0)
                st.download_button(
                    label="Download Processed Image",
                    data=img_bytes,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

elif selected_option == "Classification":
    st.markdown("""
        <style>
        .header-with-background {
            background-color: #808080; /* Background color for the header */
            color: white; /* Text color for the header */
            padding: 10px; /* Padding around the header */
            border-radius: 5px; /* Rounded corners */
            font-size: 25px; /* Font size for the header */
            font-weight: bold; /* Bold font */
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<p class='header-with-background'>Please upload the processed image to identify the type of tumor &#x2757</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = data_transform(image)
        # image = image.unsqueeze(0).to(device)

        # # Extract features using VGG16
        # with torch.no_grad():
        #     features = vgg_model(image).cpu().numpy().flatten()

        # Apply PCA transformations
        # features_pca = pca.transform([features])

        # Predict using the SVM classifier
        # prediction = svm_classifier.predict(features_pca)
        # class_labels = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}

        # # HTML and CSS to style the prediction output
        # prediction_text = f"<p style='font-size: 24px; font-weight: bold;'>Prediction: {class_labels[prediction[0]]}</p>"
        # st.markdown(prediction_text, unsafe_allow_html=True)

elif selected_option == "About":
    st.markdown("""
        <style>
        .header-with-background {
            background-color: #000000; /* Background color for the header */
            color: white; /* Text color for the header */
            padding: 10px; /* Padding around the header */
            border-radius: 5px; /* Rounded corners */
            font-size: 25px; /* Font size for the header */
            font-weight: bold; /* Bold font */
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<p class='header-with-background'>Contact</p>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        .section {
            background-color: #808080; /* Background color for the section */
            padding: 0px; /* Padding around the section */
            border-radius: 5px; /* Rounded corners */
        }
        </style>
        """, unsafe_allow_html=True)
        # Adding bullet points
    st.markdown("""
        <div class='section'>
            <ul>
                <li>&#x1F50D Ariq Dreiki Hajjanto - 5023201057</li>
                <li>Ariqdreiki213@gmail.com / 5023201057@student.its.ac.id</li>
            
        </div>
    """, unsafe_allow_html=True)
