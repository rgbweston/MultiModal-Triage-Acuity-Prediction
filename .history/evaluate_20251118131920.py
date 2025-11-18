"""
Model evaluation script for multimodal acuity classification.

This module:
- Loads preprocessed structured features, embeddings, and labels
- Splits into train/test sets
- Loads the trained multimodal neural network
- Computes predictions, softmax probabilities, accuracy, and ROC-AUC
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from train import MultiInputNet
from preprocess_features import preprocess_and_extract


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
X_struct, X_text, y, _ = preprocess_and_extract()

X_train_struct, X_test_struct, \
X_train_text, X_test_text, \
y_train, y_test = train_test_split(
    X_struct,
    X_text,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------------------
# Build Test Loader
# ---------------------------------------------------------------------
test_dataset = TensorDataset(
    torch.tensor(X_test_text, dtype=torch.float32),
    torch.tensor(X_test_struct.values, dtype=torch.float32),
    torch.tensor(y_test.values, dtype=torch.long)
)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiInputNet(
    text_dim=768,
    struct_dim=X_struct.shape[1],
    num_classes=5
)

model.load_state_dict(torch.load("rohan_multimodal_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for text_batch, struct_batch, labels in test_loader:
        text_batch, struct_batch = text_batch.to(device), struct_batch.to(device)

        outputs = model(text_batch, struct_batch)
        probs = F.softmax(outputs, dim=1)

        _, preds = torch.max(probs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
roc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

print(f"Test Accuracy: {accuracy:.2%}")
print(f"ROC AUC:       {roc:.4f}")
