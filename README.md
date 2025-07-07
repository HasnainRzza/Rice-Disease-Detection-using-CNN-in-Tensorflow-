# Paddy Doctor | Rice Disease Classification Using CNN

This project is based on the **Paddy Doctor** competition from [Kaggle](https://www.kaggle.com/competitions/paddy-disease-classification), which focuses on classifying rice leaf diseases using image data. By leveraging deep learning techniques, specifically Convolutional Neural Networks (CNNs), the project aims to automatically identify various rice plant diseases to support agricultural diagnostics and crop management.

---

## Dataset Overview

**Source**: [Kaggle - Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification)

The dataset includes:
- `train.csv`: Metadata containing image ID, rice variety, disease label, and plant age.
- `train_images/`: Directory containing the corresponding image files.
- Test data (optional for model evaluation or submission).

### Dataset Summary
- 10,407 training images
- 10 disease classes (including healthy leaves)
- 10 unique rice varieties
- Plant age range: 45 to 82 days (average ~64)
- Most common variety: `ADT45`

---

## Project Objectives

- Conduct exploratory data analysis (EDA) on rice plant metadata and images.
- Preprocess the data and encode categorical features.
- Build and train a Convolutional Neural Network for multi-class image classification.
- Evaluate model performance and visualize training results.
- Propose improvements for future development.

---

## Exploratory Data Analysis (EDA)

- Analyzed the distribution of rice varieties and age.
- Visualized sample images from both healthy and diseased plants.
- Assessed class balance and dataset structure.

---

## Data Preprocessing

- **Label Encoding**: Encoded the `label` and `variety` columns numerically using `LabelEncoder`.
- **Image Resizing**: Resized all images to `128x128` for uniformity.
- **Normalization**: Scaled pixel values to the range `[0, 1]` using `Rescaling(1./255)`.
- **Data Splitting**: Used an 80/20 training-validation split via TensorFlow’s `image_dataset_from_directory`.

---

## Model Architecture (CNN)

A custom sequential CNN model was implemented using TensorFlow with the following layers:

- Four convolutional layers with ReLU activation and max pooling.
- One dropout layer for regularization.
- A dense layer followed by a softmax output for multi-class classification.

```python
tf.keras.Sequential([
  Conv2D(128, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(16, 3, activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dropout(0.25),
  Dense(64, activation='relu'),
  Dense(num_classes, activation='softmax')
])

## Training Configuration

- **Optimizer**: Adam  
- **Loss Function**: SparseCategoricalCrossentropy (from logits)  
- **Metrics**: Accuracy  
- **Batch Size**: 16  
- **Epochs**: 100 (EarlyStopping enabled with patience of 10)  
- **Input Size**: 128x128 pixels  
- **Caching and Prefetching**: Enabled for performance optimization  

---

## Model Evaluation

The model's performance was evaluated using the following techniques:

- **Validation Accuracy and Loss**: Computed at each epoch and visualized.  
- **Early Stopping**: Monitored validation loss to prevent overfitting.  
- **Final Evaluation**: Used `model.evaluate()` on the validation set to compute overall accuracy and loss.

### Training vs Validation Loss Curve

A line chart comparing loss over epochs to monitor convergence and detect overfitting.

### Training vs Validation Accuracy Curve

A line chart to assess how well the model generalizes to unseen data.

These visualizations help determine whether the model is learning effectively and highlight opportunities for further tuning.

---

## Results and Observations

- The CNN model demonstrated stable training and validation performance due to regularization and early stopping.  
- Class distribution and image quality may impact performance, particularly for underrepresented classes.  
- Image preprocessing and consistent input sizes contributed significantly to model convergence and accuracy.

---

## Future Improvements

- Integrate data augmentation to enhance model generalization.  
- Apply transfer learning using pretrained models (e.g., MobileNetV2, EfficientNet).  
- Address class imbalance using oversampling, undersampling, or class weighting.  
- Perform hyperparameter tuning to optimize training and performance further.

---

## Getting Started

### Prerequisites

Install the required libraries:

```bash
pip install tensorflow pandas numpy seaborn matplotlib scikit-learn


Author
Muhammad Hasnain Raza
For inquiries or collaboration, feel free to connect via LinkedIn or GitHub.


For inquiries or collaboration, feel free to connect via LinkedIn or GitHub.
Viait [LinkedIn](https://www.linkedin.com/in/muhammad-hasnain-mhr/) 
