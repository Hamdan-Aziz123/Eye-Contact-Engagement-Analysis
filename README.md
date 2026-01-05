<h1 align="center">👁️ Eye Contact and Engagement Analysis</h1>
<h3 align="center"><b>Data Science Project | Fall 2025</b></h3>

<div align="center">

</div>

👥 Group Members

Name

Roll Number

Section

Hamdan Aziz

22L-6881

7A

Salman Mehmood

22L-6586

7B

📌 Project Overview

In the era of remote work and online education, maintaining participant engagement is a significant challenge due to the lack of physical non-verbal cues. Traditional methods often fail to accurately gauge attention in digital settings.

This project addresses this by developing an automated Computer Vision System that analyzes a person’s level of attention in real-time using webcam footage. By leveraging Deep Learning and geometric analysis, the system classifies user engagement into three distinct categories based on head pose and gaze direction:

🟢 Attentive: Direct eye contact with the camera/screen (Head pitch/yaw $\le 10^\circ$).

🟡 Distracted: Looking at the screen (reading/typing) but with a lowered head pose ($10^\circ < \text{Angle} \le 25^\circ$).

🔴 Disengaged: Looking away entirely, turning the head side-to-side, or closing eyes (Angle $> 25^\circ$).

🚀 Methodology & Models

We implemented and compared three distinct modeling approaches to evaluate the trade-off between geometric interpretability and raw pixel-based learning.

1️⃣ Model 1: Hand-Crafted Features (Baseline)

Approach: Traditional Machine Learning (Tabular).

Algorithm: CatBoost Classifier.

Input: Geometric features (Pitch, Vertical, Horizontal angles) extracted via preprocessing (MediaPipe).

Performance: 100% Accuracy (on geometric test set).

Why: Acts as a "Golden Standard." Since labels are derived directly from angle thresholds, this model proves that engagement is fundamentally a geometric problem.

2️⃣ Model 2: ResNet-50 (Deep Learning - CNN)

Approach: Transfer Learning with Convolutional Neural Networks.

Architecture: Pretrained ResNet-50 (ImageNet weights).

Strategy: Freeze backbone $\to$ Train Head $\to$ Fine-tune all layers.

Performance: 92.67% Accuracy.

Why: The most practical model for deployment. It works directly on raw images without needing a separate feature extractor, offering the best balance of speed and robustness.

3️⃣ Model 3: Vision Transformer (ViT-B/16)

Approach: State-of-the-Art Attention Mechanism.

Architecture: ViT-Base (patch size 16x16).

Performance: 86.93% Accuracy.

Why: Selected to test if global context attention performs better than local convolution features. While powerful, it requires more data to generalize than CNNs on this specific task.

📊 Results & Visualization

Performance Comparison

Metric

Hand-Crafted (CatBoost)

ResNet-50 (CNN)

ViT-B/16 (Transformer)

Accuracy

100.0%

92.67%

86.93%

F1-Score

1.00

0.93

0.88

Training Time

< 1 min

~15 mins

~43 mins

(Note: The Hand-Crafted model's 100% accuracy is due to the deterministic nature of the dataset's labeling process based on angles.)

📈 Visuals

Performance Comparison Plots:

Confusion Matrix (ResNet-50):

📂 Repository Structure

Eye-Contact-Engagement-Analysis/
│
├── code/
│   ├── modelC_handcrafted_ml.py       # Training script for Model 1 (CatBoost)
│   ├── Resnet50_train.py              # Training script for Model 2 (CNN)
│   ├── ViT_train.py                   # Training script for Model 3 (ViT)
│   └── 22l_6881&22l_6586.ipynb        # Main Execution Notebook (Colab)
│
├── reports/
│   ├── DS_PROJECT_REPORT.pdf          # Final Project Report (PDF)
│   ├── comparison_plots.png           # Performance Graphs
│   └── classification_metrics.png     # Metric Visualizations
│
├── outputs/                           # Training Logs & Results
│   ├── model_c/                       # CatBoost Results
│   ├── resnet50/                      # ResNet-50 Results
│   └── vit/                           # ViT Results
│
├── .gitignore                         # Files excluded from repo
├── requirements.txt                   # Python Dependencies
└── README.md                          # Project Documentation


🛠️ How to Run

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/Eye-Contact-Engagement-Analysis.git](https://github.com/YOUR_USERNAME/Eye-Contact-Engagement-Analysis.git)
cd Eye-Contact-Engagement-Analysis


2. Install Dependencies

pip install -r requirements.txt


3. Run the Training/Inference

You can run the models using the provided Jupyter Notebook (code/22l_6881&22l_6586.ipynb) in Google Colab or locally.

Alternatively, run the scripts directly:

# Train ResNet-50
python code/Resnet50_train.py --epochs 25 --batch_size 32

# Train ViT
python code/ViT_train.py --epochs 25


📚 Dataset

This project uses the Columbia Gaze Dataset (CAVE Lab, Columbia University).

Size: 5,880 Images

Subjects: 56 people (varying age, ethnicity)

Features: Annotated Head Pose (Pitch, Yaw, Roll) and Gaze Direction.

Download Dataset Here

🏆 Conclusion

While the Hand-Crafted Model demonstrated that engagement logic is geometrically deterministic, the ResNet-50 model proved to be the most practical solution for real-world deployment. It achieves high accuracy on raw images without requiring complex pre-processing pipelines to extract angles.

<div align="center">
<b>Course:</b> Data Science (CS
