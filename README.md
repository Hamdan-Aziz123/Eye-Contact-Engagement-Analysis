# 👁️ Eye Contact and Engagement Analysis

## 📌 Project Overview
**Eye Contact and Engagement Analysis** is a real-time computer vision–based Data Science project that automatically evaluates a user’s engagement level during online interactions such as virtual classes, meetings, or interviews.

The system analyzes facial orientation and visual attention to classify a person’s engagement into **three categories**:

- **Attentive** – User is looking directly at the screen or camera  
- **Distracted** – User is looking at the screen but with a lowered head posture  
- **Disengaged** – User is looking away, turning their head, or closing their eyes  

This project compares **three different AI modeling approaches**, ranging from traditional machine learning to modern deep learning and transformer-based architectures.

---

## 🎯 Objective
The primary goal of this project is to:

- Detect **eye contact and engagement level in real time**
- Compare **hand-crafted feature–based ML vs CNN vs Transformer models**
- Identify the **most robust model for deployment**
- Provide **quantitative evaluation and visual performance analysis**

---

## 🧠 Engagement Classification Logic
Engagement labels are defined using **head pose angles**:

| Engagement Level | Description | Head Pose Angle |
|------------------|------------|-----------------|
| **Attentive** | Looking directly at camera/screen | ≤ 10° |
| **Distracted** | Looking at screen but head tilted downward | 10° – 25° |
| **Disengaged** | Looking away / head turned / eyes closed | > 25° |

---

## 🏗️ Methodology & Models

### 🔹 Model 1: Hand-Crafted Features (Baseline – CatBoost)
- Uses **geometric facial features** (Pitch, Yaw, Roll angles)
- No raw images are used
- Classification is performed using **CatBoost Classifier**
- Ground-truth labels are **mathematically derived from the same angles**

✅ **Accuracy:** **100%**  
⚠️ This is considered a **“Golden Standard” baseline**, not a realistic real-world benchmark, because labels and features originate from the same source.

---

### 🔹 Model 2: ResNet-50 (CNN – Best Overall Model)
- Fine-tuned **ResNet-50** on raw facial images
- Learns features directly from pixels
- Does not require manual feature extraction
- Provides the **best balance between accuracy and robustness**

✅ **Accuracy:** **92.67%**  
🏆 **Selected as the final deployment model**

---

### 🔹 Model 3: Vision Transformer (ViT-B/16)
- Uses **patch-based image representation**
- Applies **self-attention mechanisms**
- Captures global facial context better than CNNs
- Performance limited by dataset size

✅ **Accuracy:** **86.93%**  
📌 Powerful architecture but requires larger datasets for optimal performance

---

## 📊 Model Comparison Summary

| Model | Approach | Accuracy |
|------|---------|----------|
| CatBoost | Hand-crafted features | **100%** |
| ResNet-50 | CNN (Deep Learning) | **92.67%** |
| ViT-B/16 | Transformer | **86.93%** |

---

## 📁 Dataset
- **Dataset Used:** Columbia Gaze Dataset  
- **Images:** 5,800+  
- **Subjects:** 56 individuals  
- Includes various **head poses, gaze directions, and lighting conditions**

The dataset was used to train and evaluate all three models under identical conditions.

---

## 🗂️ Project Structure

```bash
├── 22l_6881&22l_6586.ipynb        # Main execution notebook
├── modelC_handcrafted_ml.py      # CatBoost training (hand-crafted features)
├── Resnet50_train.py             # ResNet-50 training script
├── ViT_train.py                  # Vision Transformer training script
├── Outputs/
│   ├── comparison_plots.png      # Model accuracy comparison
│   ├── confusion_matrix.png      # Error visualization
│   └── DS_PROJECT_REPORT.pdf     # Complete project report
└── README.md
