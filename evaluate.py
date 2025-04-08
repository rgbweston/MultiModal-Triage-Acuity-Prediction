# evaluate.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from train import MultiInputNet, preprocess_and_extract

X_struct, X_text, y, _ = preprocess_and_extract()
_, X_test_struct, _, X_test_text, _, y_test = train_test_split(
    X_struct, X_text, y, stratify=y, test_size=0.2, random_state=42
)

test_dataset = TensorDataset(
    torch.tensor(X_test_text), torch.tensor(X_test_struct.values), torch.tensor(y_test.values)
)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiInputNet(768, X_struct.shape[1], 5)
model.load_state_dict(torch.load("rohan_multimodal_model.pth"))
model = model.to(device)
model.eval()

all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for text_batch, struct_batch, labels in test_loader:
        text_batch, struct_batch = text_batch.to(device), struct_batch.to(device)
        outputs = model(text_batch, struct_batch)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
print(f"Test Accuracy: {accuracy:.2%}")
print(f"ROC AUC: {roc:.4f}")
