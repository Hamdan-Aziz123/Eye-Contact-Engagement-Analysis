import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from timm import create_model

# ----------------------------
# CONFIGURATION
# ----------------------------
BEST_MODEL_PATH = "/content/drive/MyDrive/ResNet_Project/outputs_vit/best_vit.pth"
TEST_DIR = "/content/drive/MyDrive/ResNet_Project/dataset/test"   # <-- change if needed
BATCH_SIZE = 32
NUM_CLASSES = 3
# ----------------------------

# Transforms used during training
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
report_text = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

print("\n===== Classification Report =====\n")
print(report_text)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n===== Confusion Matrix =====\n")
print(cm)

# Save Report JSON
save_path = "/content/drive/MyDrive/ResNet_Project/outputs_vit/classification_report_vit.json"
with open(save_path, "w") as f:
    json.dump(report, f, indent=4)

# Save text report
save_path_txt = "/content/drive/MyDrive/ResNet_Project/outputs_vit/classification_report_vit.txt"
with open(save_path_txt, "w") as f:
    f.write(report_text)

print("\nSaved classification report successfully!")
