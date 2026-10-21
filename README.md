# CIFAR-10 Image Classification Using Deep Learning

An end-to-end deep learning project that classifies images into 10 categories using PyTorch, compares three CNN architectures, evaluates model performance, and provides an interactive Streamlit web application.

## Project Overview

This project explores image classification using the CIFAR-10 dataset. It covers the complete machine learning workflow, from exploratory data analysis and preprocessing to model training, evaluation, explainability, and deployment.

Three models are compared: a baseline CNN, an improved CNN, and a pretrained ResNet-18. The selected model is integrated into a Streamlit application that allows users to upload an image and view the predicted class and confidence scores.

## Key Features

- Exploratory data analysis and image visualization
- Image normalization and data augmentation
- Baseline CNN implemented using PyTorch
- Improved CNN with batch normalization and dropout
- Transfer learning using pretrained ResNet-18
- Training and validation loss and accuracy curves
- Test-set evaluation and classification reports
- Confusion matrix and per-class accuracy analysis
- Grad-CAM visualizations for model explainability
- Interactive Streamlit application with image upload and top-5 predictions

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 color images of size 32 × 32 pixels across 10 classes.

- Training images: 50,000
- Test images: 10,000
- Number of classes: 10

The classes are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

The dataset is loaded using `torchvision.datasets.CIFAR10`.

## Models Implemented and Their Architectures

Three CNN-based models were implemented and evaluated on CIFAR-10 to compare baseline performance, architectural improvements, and transfer learning.

### 1. Baseline CNN

A compact convolutional neural network that establishes baseline classification performance using two convolutional layers and three fully connected layers.

**Architecture:**

`Input (3×32×32) → Conv2D (3→6, 5×5) → ReLU → MaxPool → Conv2D (6→16, 5×5) → ReLU → MaxPool → Flatten (400) → FC (400→120) → ReLU → FC (120→84) → ReLU → FC (84→10)`

### 2. Improved CNN

A deeper CNN designed to improve feature extraction and generalization. It incorporates batch normalization, additional convolutional layers, and dropout regularization.

**Architecture:**

`Input → Conv2D (3→32) → BatchNorm → ReLU → Conv2D (32→32) → BatchNorm → ReLU → MaxPool → Dropout → Conv2D (32→64) → BatchNorm → ReLU → Conv2D (64→64) → BatchNorm → ReLU → MaxPool → Dropout → Flatten (4096) → FC (4096→256) → ReLU → Dropout → FC (256→10)`

### 3. Pretrained ResNet-18

A ResNet-18 model initialized with pretrained ImageNet weights and fine-tuned for CIFAR-10 classification. Its initial convolution is adapted for 32×32 images, the initial max-pooling layer is removed, and the final fully connected layer is replaced with a 10-class classifier.

**Architecture:**

`Input (3×32×32) → Adapted Conv2D → Residual Layer 1 → Residual Layer 2 → Residual Layer 3 → Residual Layer 4 → Adaptive Average Pooling → FC (512→10)`

### Training Configuration

- **Loss function:** Cross-Entropy Loss
- **Optimizers:** Adam for the custom CNNs; Adam with separate learning rates for the ResNet backbone and classifier
- **Data augmentation:** Random cropping and horizontal flipping during training
- **Model selection:** Validation performance
- **Final evaluation:** Held-out CIFAR-10 test set


## Model Evaluation

All three models were evaluated on the official CIFAR-10 test set. The recorded results are:

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| Baseline CNN | 0.8303 | 70.98% |
| Improved CNN | 0.4186 | 85.66% |
| Pretrained ResNet-18 | 0.2674 | 92.12% |

ResNet-18 achieved the highest test accuracy of **92.12%** among the three models. It was selected for deployment based on validation performance.

## Results and Visualizations

The `outputs/` directory contains visualizations and CSV files generated during the experiments.

| File | Description |
|---|---|
| `class_distribution.png` | Distribution of CIFAR-10 classes |
| `pixel_distribution.png` | Pixel intensity analysis |
| `training_curves.png` | Baseline CNN training and validation curves |
| `model_comparison_accuracy.png` | Baseline and improved CNN comparison |
| `three_model_accuracy_comparison.png` | Validation accuracy comparison across all models |
| `final_test_accuracy_comparison.png` | Final test accuracy comparison |
| `three_model_confusion_matrices.png` | Confusion matrices for the three models |
| `per_class_accuracy_comparison.png` | Per-class accuracy comparison |
| `resnet_gradcam_examples.png` | Grad-CAM explanations for ResNet-18 predictions |

The CSV files provide detailed model metrics, classification reports, and per-class accuracy results.

## Streamlit Web Application

The application provides an interactive interface for image classification.

Users can:

1. Upload a JPG, JPEG, or PNG image.
2. View the uploaded image.
3. Obtain a predicted CIFAR-10 class.
4. View the prediction confidence.
5. Explore the top-five predicted classes and their confidence scores.

**Note:** The model is trained on small CIFAR-10 images. Predictions on unrelated real-world images or images outside the 10 supported categories may be unreliable. Confidence scores do not guarantee correctness.

## Project Structure

```text
cifar10-image-classifier/
├── CIFAR10_Image_Classification.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── deployment_model.pth
│   ├── deployment_config.json
│   ├── best_cifar10_cnn.pth
│   └── best_improved_cifar10_cnn.pth
└── outputs/
    ├── class_distribution.png
    ├── pixel_distribution.png
    ├── training_curves.png
    ├── final_test_comparison.csv
    ├── final_test_accuracy_comparison.png
    ├── three_model_confusion_matrices.png
    ├── per_class_accuracy.csv
    ├── per_class_accuracy_comparison.png
    └── resnet_gradcam_examples.png
```

The large ResNet-18 training checkpoint is not included in the repository ZIP. The deployment checkpoint contains the selected model's weights.

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-image-classifier.git
cd cifar10-image-classifier
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell, activate it using:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Deployment

The application is intended for deployment using [Streamlit Community Cloud](https://share.streamlit.io/).

Once the project is pushed to GitHub:

1. Sign in to Streamlit Community Cloud using GitHub.
2. Create a new app and select this repository.
3. Select the `main` branch.
4. Set `app.py` as the main file.
5. Deploy the application.

**Live Demo:** [Classifier](https://nagarjun622-cifar10-image-classifier-app-y4fsre.streamlit.app/)

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Future Improvements

- Integrate Grad-CAM directly into the web application.
- Add downloadable prediction reports.
- Experiment with additional CNN architectures and augmentation techniques.
- Track experiments and model versions.
- Improve the application interface and inference feedback.
