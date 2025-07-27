
# Convolutional Neural Networks for Image Classification

This repository contains a complete implementation of Convolutional Neural Networks (CNNs) from scratch using both NumPy and TensorFlow/Keras for classifying handwritten digits from the MNIST dataset. The project was submitted as part of the final course assignment and comprises multiple stages: manual implementation of CNN components, building deep learning models, and visualizing learned features.

---

## 📁 Files

- `Code1.py` — Basic Neural Network concepts and manual forward propagation.
- `Code2.py` — Implementation of Convolution and Pooling layers using NumPy.
- `Code3.py` — End-to-end CNN pipeline using TensorFlow/Keras.
- `23B2158_Final Report.pdf` — Detailed report of the project with learnings, challenges, and results.

---

## 📌 Project Sections

### 1. Data Loading & Preprocessing
- MNIST dataset loaded using Keras.
- Input images reshaped to (28, 28, 1) for compatibility with Conv2D.
- Pixel values normalized to [0, 1].

### 2. Manual Convolution & Pooling
- Performed convolution on a single MNIST image using a 5x5 filter.
- Applied 2×2 max pooling.
- Visualized the original, convolved, and pooled feature maps.
- Implemented zero-padding, stride, and pooling manually using NumPy.

### 3. CNN Model Architecture
Implemented using TensorFlow/Keras:
- `Conv2D`: 16 filters (5x5), ReLU activation.
- `MaxPooling2D`: 2x2.
- `Conv2D`: 32 filters (5x5), ReLU activation.
- `MaxPooling2D`: 2x2.
- `Flatten` → `Dense(64)` → `Dense(10, softmax)`.

### 4. Visualizing Feature Maps
- Visualized intermediate activations of each layer.
- Showed how early layers learn edges, while deeper layers detect textures/patterns.

### 5. Model Training & Evaluation
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Trained for 5 epochs with batch size 128.
- Plotted training/validation accuracy and loss.
- Evaluated using a confusion matrix and classification report.

---

## 📘 Learnings

- Neural Network components: layers, activations (ReLU, Softmax), and backpropagation.
- Mathematical intuition behind convolution and pooling.
- Building CNNs using both NumPy and high-level frameworks.
- Feature extraction and interpretation using layer visualizations.
- Model generalization via dropout and data augmentation.

---

## 🚧 Challenges & Solutions

- **Training Time**: Resolved by using Google Colab GPU.
- **Overfitting**: Resolved using dropout and image augmentation.
- **Hyperparameter Tuning**: Adjusted filters, learning rates, and batch sizes iteratively.

---

## 🔍 Key Findings

- CNNs are powerful for image tasks due to hierarchical feature extraction.
- Visualization shows how CNNs build understanding from edges to digits.
- Overfitting increases with network depth — dropout helps mitigate.
- Feature extraction from CNNs is reusable via transfer learning.

---

## 🔮 Future Work

- Implement deeper models like **ResNet**, **VGG**, or **MobileNet**.
- Explore **Transfer Learning** on medical or color image datasets.
- Deploy model using **TensorFlow Lite** or build a web interface.
- Investigate **Vision Transformers (ViTs)** for performance comparisons.

---

## 👨‍💻 Author

**Santosh Guntuku**  
Roll No: `23B2158`  
Department of Mechanical Engineering, IIT Bombay

---

## 📜 License

This project is for educational purposes. Feel free to fork and build upon it!

