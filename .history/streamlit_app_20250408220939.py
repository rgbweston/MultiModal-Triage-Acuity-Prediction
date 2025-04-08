import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
import os

# Page config
st.set_page_config(page_title="Patient Acuity Predictor", layout="wide")

# --- Style ---
st.markdown("""
<style>
.reportview-container {
    background: #f8f9fa;
}
.sidebar .sidebar-content {
    background: #e9ecef;
}
.stButton>button {
    background-color: #2E86C1;
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# --- Paths (relative for GitHub) ---
transformer_model_path = "sentence-transformers/all-mpnet-base-v2"
model_path = "models/rohan_multimodal_model.pth"
scaler_path = "models/rohan_multi_scaler.pkl"
banner_path = "assets/ucl_banner.png"

# --- Model class ---
class MultiInputNet(nn.Module):
    def __init__(self, text_seq_len, text_in_channels, num_structured_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 5, padding=1)
        self.pool = nn.MaxPool1d(3, 2, 1)
        self.text_branch_output_size = 32 * 191
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
        return self.classifier(torch.cat((x_text, x_struct), dim=1))

# --- Load resources ---
@st.cache_resource
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
    model = AutoModel.from_pretrained(transformer_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.to(device), device

@st.cache_resource
def load_model():
    model = MultiInputNet(768, 1, 14, 5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

# --- Banner ---
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.title("Predicting Acuity of a Patient")

# --- Disclaimer ---
st.warning("This tool is a proof-of-concept and not intended for clinical decision-making.")

# --- Sidebar Input ---
st.sidebar.title("Vitals")
unit = st.sidebar.radio("Temperature Unit", ["Fahrenheit", "Celsius"], index=0)
temp = st.sidebar.number_input("Temperature", 32.0, 120.0, 98.6)
hr = st.sidebar.number_input("Heart Rate", 0, 200, 70)
rr = st.sidebar.number_input("Resp Rate", 0, 100, 16)
o2 = st.sidebar.number_input("O2 Sat", 0, 100, 98)
sbp = st.sidebar.number_input("SBP", 0, 300, 120)
dbp = st.sidebar.number_input("DBP", 0, 200, 80)
pain = st.sidebar.number_input("Pain", 0, 10, 0)

arrival = st.sidebar.selectbox("Arrival Mode", ["AMBULANCE", "HELICOPTER", "OTHER", "UNKNOWN", "WALK IN"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
complaint = st.sidebar.text_input("Chief Complaint", "")

# --- Prepare inputs ---
temp_f = temp * 9/5 + 32 if unit == "Celsius" else temp
numeric = np.array([[temp_f, hr, rr, o2, sbp, dbp, pain]])
scaler = load_scaler()
scaled_numeric = scaler.transform(numeric).flatten()

arrival_dict = {
    "AMBULANCE": [1, 0, 0, 0, 0],
    "HELICOPTER": [0, 1, 0, 0, 0],
    "OTHER": [0, 0, 1, 0, 0],
    "UNKNOWN": [0, 0, 0, 1, 0],
    "WALK IN": [0, 0, 0, 0, 1],
}
gender_dict = {
    "Female": [1, 0],
    "Male": [0, 1],
}
structured = np.array(scaled_numeric.tolist() + arrival_dict[arrival] + gender_dict[gender], dtype=np.float32)
structured_tensor = torch.tensor(structured).view(1, -1)

# --- Text embedding ---
tokenizer, transformer, device = load_transformer()
if complaint:
    tokens = tokenizer(complaint, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        emb = transformer(**tokens).last_hidden_state.mean(dim=1).cpu()
else:
    emb = torch.zeros((1, 768))

# --- Predict ---
if st.button("Predict"):
    model = load_model()
    output = model(emb, structured_tensor)
    prediction = torch.argmax(output, dim=1).item() + 1
    st.subheader(f"Predicted ESI Level: {prediction}")