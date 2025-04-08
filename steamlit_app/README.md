# Acuity Prediction – Streamlit App

This Streamlit app predicts a patient’s Emergency Severity Index (ESI) level (1–5) using both vital signs and the chief complaint. It uses a trained multimodal deep learning model that combines structured clinical data and MPNet-based text embeddings.

## How to run (in the TRE)

To launch the app from the TRE terminal, run the following command:

docker run --rm --privileged -p 8501:8501
-v /home/rmhirgb/data:/workspace/data
image4
streamlit run /workspace/data/streamlit.py
--server.port=8501 --server.address=0.0.0.0


Once running, open a browser and go to: http://localhost:8501


## Instructions for use

1. Enter patient vital signs in the sidebar:
   - Temperature (select either Celsius or Fahrenheit)
   - Heart rate
   - Respiratory rate
   - Oxygen saturation
   - Systolic and diastolic blood pressure
   - Pain score (0 to 10)

2. Select the patient's method of arrival (e.g. ambulance, walk-in) and gender.

3. Enter the chief complaint as free text (e.g. "shortness of breath").

4. Click the "Predict" button to receive the predicted ESI level and suggested clinical actions based on acuity.

## Important note

This app is a prototype and is intended for demonstration purposes only. It is not approved for clinical use, and all decisions should be made by qualified healthcare professionals.

Model weights and scalers are not provided in this repository due to restrictions associated with the MIMIC data use agreement, which prohibits sharing derived artefacts trained on the dataset.

