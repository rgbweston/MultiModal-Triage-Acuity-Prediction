"""
Feature preprocessing pipeline for triage and ED stays data.

This module:
- Merges the triage and edstays datasets
- Cleans and validates numeric fields (e.g., pain, acuity)
- One-hot encodes categorical variables
- Scales numeric features with StandardScaler
- Generates dense text embeddings for chief complaints using MPNet
- Caches embeddings to disk (embeddings.npy) to avoid recomputation

Returns structured numeric/categorical features, text embeddings,
targets, and the fitted scaler needed for inference-time preprocessing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
import joblib

# Mpnet base model: not clinically specialized but good general performance
transformer_path = 'models--sentence-transformers--all-mpnet-base-v2'


def batch_generate_embeddings(text_list, tokenizer, model, device, batch_size=32):
    """
    Generate transformer embeddings for a list of texts in batches.

    Batching helps avoid GPU/CPU memory exhaustion. Each batch is tokenized,
    encoded through the MPNet model, and mean-pooled across the sequence length.

    Args:
        text_list (list[str]): Texts to embed.
        tokenizer: Pretrained tokenizer.
        model: Pretrained transformer model.
        device: Torch device (CPU or GPU).
        batch_size (int): Number of samples per batch.

    Returns:
        np.ndarray: Array of embeddings.
    """
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size].tolist()

        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        with torch.no_grad():
            output = model(**tokens)
            batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()

        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def preprocess_and_extract():
    """
    Preprocess triage and ED stays data and extract structured features,
    text embeddings, targets, and the fitted numeric scaler.

    Steps:
        1. Load CSVs and merge on stay_id
        2. Drop unused identifiers and rare fields
        3. Validate pain and acuity ranges
        4. One-hot encode categorical variables
        5. Scale numeric columns with StandardScaler
        6. Generate MPNet embeddings (cached to embeddings.npy)

    Returns:
        X_structured (pd.DataFrame): Numeric and one-hot encoded features.
        embeddings (np.ndarray): Dense embedding matrix.
        y (pd.Series): Zero-indexed acuity class labels.
        scaler (StandardScaler): Fitted scaler for inference use.
    """
    triage = pd.read_csv("triage.csv.gz")
    edstays = pd.read_csv("edstays.csv.gz")

    df = pd.merge(edstays, triage, on="stay_id", how="inner")

    df = df.drop(columns=[
        "subject_id_x", "stay_id", "hadm_id", "intime", "outtime",
        "race", "disposition", "subject_id_y"
    ])
    df = df.dropna()

    df = pd.get_dummies(df, columns=["arrival_transport", "gender"])

    df = df[df["pain"].apply(lambda x: str(x).isnumeric())]
    df = df[df["pain"].apply(lambda x: 0 <= int(x) <= 10)]
    df = df[df["acuity"].apply(lambda x: 0 <= int(x) <= 5)]

    scaler = StandardScaler()
    numeric_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X_structured = df.drop(columns=["acuity", "chiefcomplaint"]).astype(np.float32)
    y = df["acuity"].astype(int) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModel.from_pretrained(transformer_path).to(device)

    embeddings_file = "embeddings.npy"
    if not os.path.exists(embeddings_file):
        print("Generating embeddings...")
        embeddings = batch_generate_embeddings(df["chiefcomplaint"], tokenizer, model, device)
        np.save(embeddings_file, embeddings)
    else:
        embeddings = np.load(embeddings_file)

    return X_structured, embeddings, y, scaler


# Execute preprocessing and save the scaler
if __name__ == "__main__":
    X_structured, embeddings, y, scaler = preprocess_and_extract()
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")
