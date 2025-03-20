# Computer Vision Edge Detection & Segmentation WebApp

## Overview
This web application utilizes [Streamlit](https://streamlit.io/) to perform multiple edge detection and image segmentation techniques. Additionally, it leverages deep learning algorithms for facial detection. Simply upload your image, and the app instantly provides visual outputs demonstrating edge detection, image segmentation, and facial recognition results.

## Features
- **Edge Detection**:
  - Sobel
  - Scharr
  - Prewitt
  - Canny
  - Laplacian

- **Image Segmentation**:
  - Otsu Thresholding
  - K-Means Clustering

- **Face Detection**:
  - MTCNN (Multi-task Cascaded Convolutional Networks)

## Technologies Used
- [Streamlit](https://streamlit.io/) for interactive web application interface
- OpenCV for image processing
- NumPy and Matplotlib for numerical operations and visualization
- scikit-image and scikit-learn for segmentation and clustering techniques
- facenet-pytorch and CVLib for face detection

## Installation and Usage

1. **Clone the repository:**
```bash
git clone https://github.com/azizzoaib786/computer-vision-edge-detection-webapp-streamlit.git
cd computer-vision-edge-detection-webapp-streamlit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Access the webapp:**
Navigate to the provided URL (`localhost:8501`) in your browser.

## Deployment
You can deploy this app directly using Streamlit Cloud:

- Push your repository to GitHub
- Visit [Streamlit Cloud](https://streamlit.io/cloud)
- Connect your repository and deploy the application in minutes!

## Contributing
Feel free to fork this repository, make your enhancements, and submit a pull request.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.