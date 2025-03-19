import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from sklearn.cluster import KMeans
import cvlib as cv

st.set_page_config(layout="wide")
st.title("Facial Edge Detection and Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(image_rgb, use_column_width=True)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gaussian_blurred = cv2.GaussianBlur(gray, (21, 21), 4)

    # Edge Detection Functions
    def edge_detection(img, kernel_x, kernel_y):
        grad_x = cv2.filter2D(img, -1, kernel_x)
        grad_y = cv2.filter2D(img, -1, kernel_y)
        magnitude = np.hypot(grad_x, grad_y)
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        return magnitude

    edge_methods = {
        "Sobel": (np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), np.array([[-1,-2,-1],[0,0,0],[1,2,1]])),
        "Scharr": (np.array([[-3,0,3],[-10,0,10],[-3,0,3]]), np.array([[-3,-10,-3],[0,0,0],[3,10,3]])),
        "Prewitt": (np.array([[-1,0,1],[-1,0,1],[-1,0,1]]), np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
    }

    edge_results = {}
    for name, (kx, ky) in edge_methods.items():
        edge_results[name] = edge_detection(gaussian_blurred, kx, ky)

    edge_results["Canny"] = cv2.Canny(gaussian_blurred, 20, 60)

    # Manual Laplacian Edge Detection Kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian_edges = cv2.filter2D(gaussian_blurred, cv2.CV_64F, laplacian_kernel)
    laplacian_edges = np.uint8(np.absolute(laplacian_edges))
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
    edge_results["Laplacian"] = laplacian_edges

    st.subheader("Edge Detection Results")
    fig, axes = plt.subplots(1, len(edge_results), figsize=(20, 5))
    for ax, (name, result) in zip(axes, edge_results.items()):
        ax.imshow(result, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    st.pyplot(fig)

    # Segmentation
    st.subheader("Segmentation Results")

    _, otsu_thresh = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image_rgb.shape).astype(np.uint8)

    # MTCNN Face Detection
    detector = MTCNN()
    boxes, probs, landmarks = detector.detect(image_rgb, landmarks=True)
    mtcnn_img = image_rgb.copy()
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(mtcnn_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(mtcnn_img, f'{prob:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    segmentation_results = {
        "Otsu Threshold": otsu_thresh,
        "K-Means Segmentation": segmented_img,
        "MTCNN Detection": mtcnn_img,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, seg_img) in zip(axes, segmentation_results.items()):
        if seg_img.ndim == 2:
            ax.imshow(seg_img, cmap='gray')
        else:
            ax.imshow(seg_img)
        ax.set_title(title)
        ax.axis("off")

    st.pyplot(fig)