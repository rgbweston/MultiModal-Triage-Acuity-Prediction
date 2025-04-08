# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle

from preprocess_features import preprocess_and_extract

class MultiInputNet(nn.Module):
    def __init__(self, text_dim, num_structured_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 5, padding=1)
        self.pool = nn.MaxPool1d(2, stride=2, padding=1)
        self.text_branch_output_size = 32 * 384
        self.structured_fc = nn.Sequential(
            nn.Linear(num_structured_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.text_branch_output_size + 16, num_classes)

    def forward(self, text_input, structured_input):
        if text_input.dim() == 2:
            text_input = text_input.unsqueeze(1)
        x_text = F.relu(self.conv1(text_input))
        x_text = self.pool(x_text)
        x_text = F.relu(self.conv2(x_text))
        x_text = self.pool(x_text)
        x_text = x_text.view(x_text.size(0), -1)
        x_struct = self.structured_fc(structured_input)
        x = torch.cat((x_text, x_struct), dim=1)
        return self.classifier(x)

X_struct, X_text, y, scaler = preprocess_and_extract()
X_train_struct, X_test_struct, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_struct, X_text, y, stratify=y, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(
    torch.tensor(X_train_text), torch.tensor(X_train_struct.values), torch.tensor(y_train.values)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiInputNet(768, X_struct.shape[1], 5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for text_batch, struct_batch, labels in train_loader:
        text_batch, struct_batch, labels = text_batch.to(device), struct_batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(text_batch, struct_batch), labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "rohan_multimodal_model.pth")
with open("rohan_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
