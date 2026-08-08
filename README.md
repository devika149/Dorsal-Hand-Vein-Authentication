# 🖐️ Dorsal Hand Vein Authentication

A secure, contactless biometric authentication system that verifies user identity using **dorsal hand vein patterns**. Built with a hybrid **CNN + SVM** deep learning pipeline, the system leverages infrared imaging and advanced image processing to deliver high-accuracy, forgery-resistant authentication.

---

## 📌 Overview

Traditional authentication methods like passwords, PINs, and even surface-level biometrics (fingerprints, facial recognition) are vulnerable to spoofing, forgery, and environmental variation. **Dorsal hand vein patterns**, being subcutaneous, offer a highly secure alternative — they can't be seen, copied, or easily replicated.

This project implements a **Python-based dorsal hand vein authentication system** that combines:
- **CNN (Convolutional Neural Network)** for automatic, deep feature extraction from vein images
- **SVM (Support Vector Machine)** for robust, accurate classification

The result is a system with improved recognition accuracy and significantly reduced **False Acceptance Rate (FAR)** and **False Rejection Rate (FRR)**.

---

## ✨ Key Features

- 🔒 **High Security** — Vein patterns are internal and virtually impossible to forge
- 🖐️ **Contactless Authentication** — No physical contact required, improving hygiene and convenience
- 🧠 **Hybrid Deep Learning** — CNN for feature extraction + SVM for classification
- 🖼️ **Advanced Preprocessing** — ROI extraction, CLAHE enhancement, noise reduction
- 📊 **Performance Metrics** — Evaluated using Accuracy, FAR, and FRR
- 🛡️ **Spoof Protection** — Dedicated hand-detection validation before processing
- 👥 **Admin Panel** — Manage registered users, view authentication logs

---

## 🏗️ System Architecture

The system follows a modular pipeline:

```
Capture / Upload Vein Image
        ↓
  Grayscale Conversion
        ↓
    ROI Extraction
        ↓
   CLAHE Enhancement
        ↓
    Noise Reduction
        ↓
 CNN Feature Extraction
        ↓
   SVM Classification
        ↓
Access Granted / Denied
```

### Modules
| Module | Responsibility |
|---|---|
| **Image Acquisition** | Captures/loads dorsal hand vein images (IR imaging) |
| **Image Preprocessing** | Grayscale conversion, ROI extraction, noise reduction, CLAHE |
| **Feature Extraction (CNN)** | Learns deep, discriminative vein features |
| **Classification (SVM)** | Classifies feature vectors for identity matching |
| **Authentication & Evaluation** | Final decision-making and logging |

---

## 🛠️ Tech Stack

**Language:** Python

**Core Libraries:**
- **OpenCV** — Image preprocessing (grayscale, ROI, CLAHE, noise reduction)
- **TensorFlow / Keras** — CNN model for feature extraction
- **Scikit-learn** — SVM classifier and performance evaluation
- **NumPy** — Numerical and matrix operations
- **Matplotlib** — Visualization of results and metrics

**Development Tools:** Jupyter Notebook, Visual Studio Code

**Hardware Requirement:** Infrared (IR) imaging device / vein scanner for capturing subcutaneous vein patterns

---

## ⚙️ How It Works

1. **Registration** — User uploads/captures left and right hand vein images, which are stored as biometric templates
2. **Preprocessing** — Images are converted to grayscale, the region of interest is extracted, and CLAHE enhances vein visibility
3. **Feature Extraction** — A CNN model extracts a deep feature vector representing the unique vein structure
4. **Matching** — The extracted feature vector is compared against stored templates using cosine similarity / SVM classification
5. **Decision** — If similarity exceeds the threshold, access is granted; otherwise, it's denied

---

## 📊 Performance Metrics

The system is evaluated using standard biometric metrics:

- **Accuracy** = (Correct Predictions / Total Predictions) × 100
- **False Acceptance Rate (FAR)** = False Acceptances / Total Impostor Attempts
- **False Rejection Rate (FRR)** = False Rejections / Total Genuine Attempts

The hybrid CNN–SVM approach significantly outperforms traditional single-classifier systems in reducing FAR and FRR.

---

## 🚀 Applications

- 🏦 Banking & financial security
- 🏥 Healthcare access systems
- 🏢 Corporate & government security
- 🎖️ Military access control

