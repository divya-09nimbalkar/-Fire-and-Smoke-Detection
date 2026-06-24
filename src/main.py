import streamlit as st
import os
from inference import predict_image
from config import CLASS_NAMES

st.title("🔥 Fire & Smoke Detection Dashboard")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict_image("temp.jpg")
    st.success(f"Prediction: {prediction.upper()}")

# ✅ Evaluate test set
if st.button("Evaluate Test Set"):
    test_dir = "data/processed/test"
    correct, total = 0, 0
    for cls in CLASS_NAMES:
        folder = os.path.join(test_dir, cls)
        for img in os.listdir(folder):
            pred = predict_image(os.path.join(folder, img))
            if pred == cls:
                correct += 1
            total += 1
    st.info(f"Test Accuracy: {correct}/{total} = {correct/total:.2%}")
