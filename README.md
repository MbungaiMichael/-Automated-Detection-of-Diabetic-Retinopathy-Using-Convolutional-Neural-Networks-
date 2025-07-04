
# Automated Detection of Diabetic Retinopathy Using Convolutional Neural Networks

## Project Overview

Diabetic Retinopathy (DR) is one of the leading causes of blindness in working-age adults worldwide. It is caused by prolonged high blood sugar levels that damage the blood vessels in the retina. Early detection and timely treatment of DR are critical to preventing vision loss. However, manual screening is time-consuming, costly, and highly dependent on the skill and experience of ophthalmologists.

This project aims to develop an automated, deep learning-based system utilizing Convolutional Neural Networks (CNNs) to detect and classify stages of diabetic retinopathy from retinal fundus images. By leveraging CNN’s powerful feature extraction capabilities, the model can learn to identify microaneurysms, hemorrhages, and other DR-related lesions directly from raw images, eliminating the need for manual feature engineering.

## Objectives

- Automate the detection of Diabetic Retinopathy using CNN
- Classify images into five DR severity stages:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR
- Achieve high accuracy and sensitivity for clinical relevance
- Provide explainable visual outputs (e.g., Grad-CAM)


## Model Architecture

- Input Layer: Resized RGB retinal images ( 224x224x3)
- Convolutional Layers (Conv2D + ReLU)
- MaxPooling Layers (2*2)
- Dropout Layers for regularization
- Fully Connected Dense Layers
- Output Layer: Softmax activation for multi-class classification


## Dataset

- **Source**: [Diabetic retinopathy Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)
- **Images**: Retinal fundus photographs
- **Labels**: Multi-class (0 to 4) based on DR severity
- **project**: [Project here](https://github.com/MbungaiMichael/Automated-Detection-of-Diabetic-Retinopathy-Using-Convolutional-Neural-Networks/blob/main/Diabetic%20Retinopathy.ipynb)


## Workflow

1. **Data Preprocessing**:
   - Image resizing, normalization
   - Data augmentation (rotation, zoom, flip)
   - Contrast enhancement

2. **Model Building**:
   - CNN was designed from scratch 
   - Compiled with `categorical_crossentropy` loss and `Adam` optimizer of learning rate 0.0001

3. **Training & Validation**:
   - Split the dataset into training (80%), validation sets (20%) 
   - Applied **Dropout (rate = 0.2)** between dense layers to reduce overfitting and improve generalization.

4. **Evaluation**:
   - Metrics: Accuracy

5. **Visualization**:
   - Training/Validation accuracy and loss curves


## Tech Stack

| Tool/Library      | Description                       |
|-------------------|-----------------------------------|
| Python            | Core programming language         |
| TensorFlow/Keras  | Deep learning framework           |
| OpenCV            | Image preprocessing               |
| NumPy & Pandas    | Data manipulation                 |
| Matplotlib/Seaborn| Plotting and visualization        |
| Scikit-learn      | Evaluation metrics                |
| Jupyter Notebook  | Interactive development           |


## Results

| Metric             | Value    |
|--------------------|----------|
| Training Accuracy  | 92.79%   |
| Test Accuracy      | 93.8%    |

The model achieved strong performance with generalization capability, making it suitable for real-world screening applications.

