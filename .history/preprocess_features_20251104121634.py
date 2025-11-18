# preprocess_features.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch

transformer_path = 'models--sentence-transformers--all-mpnet-base-v2'

def batch_generate_embeddings(text_list, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size].tolist()
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

def preprocess_and_extract():
    triage = pd.read_csv('triage.csv.gz')
    edstays = pd.read_csv('edstays.csv.gz')
    df = pd.merge(edstays, triage, on='stay_id', how='inner')

    df = df.drop(columns=['subject_id_x', 'stay_id', 'hadm_id', 'intime', 'outtime', 'race', 'disposition', 'subject_id_y'])
    df = df.dropna()
    df = pd.get_dummies(df, columns=['arrival_transport', 'gender'])

    df = df[df['pain'].apply(lambda x: str(x).isnumeric())]
    df = df[df['pain'].apply(lambda x: 0 <= int(x) <= 10)]
    df = df[df['acuity'].apply(lambda x: 0 <= int(x) <= 5)]

    scaler = StandardScaler()
    numeric_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X_structured = df.drop(columns=['acuity', 'chiefcomplaint']).astype(np.float32)
    y = df['acuity'].astype(int) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModel.from_pretrained(transformer_path).to(device)

    embeddings_file = "embeddings.npy"
    if not os.path.exists(embeddings_file):
        print("Generating embeddings...")
        embeddings = batch_generate_embeddings(df['chiefcomplaint'], tokenizer, model, device)
        np.save(embeddings_file, embeddings)
    else:
        embeddings = np.load(embeddings_file)

    return X_structured, embeddings, y, scaler
