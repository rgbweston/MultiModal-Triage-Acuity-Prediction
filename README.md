# MultiModal-Triage-Acuity-Prediction

# Acuity Prediction Using Multimodal Deep Learning

This repository contains all code and resources for a project that predicts patient acuity levels (Emergency Severity Index, ESI 1–5) using both structured clinical data and free-text chief complaints. The model combines traditional clinical features with MPNet-generated text embeddings using a multimodal convolutional neural network.

## Project Overview

- **Objective:** Predict triage acuity levels to assist emergency department prioritisation.
- **Data source:** MIMIC-ED (physionet.org)
- **Approach:** Deep learning with structured + text input, built and evaluated in PyTorch.
- **Deployment:** Streamlit-based web app with Docker support, designed for use within a Trusted Research Environment (TRE).

---

## How to Run the Streamlit App (TRE only)

The app is deployed via Docker and can be run from a TRE terminal using:

docker run --rm --privileged -p 8501:8501
-v /home/rmhirgb/data:/workspace/data
image4
streamlit run /workspace/data/streamlit.py
--server.port=8501 --server.address=0.0.0.0


Once running, open a browser and go to: http://localhost:8501


Follow the on-screen instructions to input patient information and predict acuity level.

---

## Repo structure

```
├── all-mpnet-base-v2/
│
├── TRE_scripts/
│   ├── docker-compose.yml
│   └── Dockerfile
│
├── Dockerfile
├── requirements.txt
│
├── mimic-iv-ed-2.2-DEMO/
│   ├── diagnosis_demo.csv
│   ├── edstays_demo.csv
│   ├── pyxis_demo.csv
│   ├── triage_demo.csv
│   └── vitalsign_demo.csv
│
├── streamlit_app/
│   ├── streamlit_app.py
│   └── streamlit_demo.py
│
├── evaluate.py
├── preprocess_features.py
├── train.py
│
├── model_results/
│
└── README.md
```
---

## Notes

- Model weights and scalers are **not provided in this repository** due to restrictions associated with the MIMIC data use agreement, which prohibits sharing derived artefacts trained on the dataset.
- The app and code are provided for educational and demonstration purposes only.
- This project is not intended for clinical use.

---

## License

This repository is shared under the MIT License. Use of MIMIC data is governed separately under the PhysioNet credentialed data use agreement.




