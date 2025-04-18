import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import base64

# Load the saved KNN model with error handling
try:
    with open(r'C:\Users\hp\OneDrive\Desktop\my project\cancer prediction\KNN_7_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Use a new scaler for input data (since scaler.pkl is missing)
scaler = StandardScaler()

# Load background image safely with transparency overlay

def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        bg_image = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        h1, h2, h3, p, div, label {{
            color: #FFFFFF;
        }}
        </style>
        """
        st.markdown(bg_image, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load background image: {e}")

set_background(r'C:\Users\hp\OneDrive\Desktop\my project\cancer prediction\pngtree-cancer-cells-and-the-red-cell-picture-image_2762482.jpg')

# Title and description
st.title("🩺 Advanced Breast Cancer Prediction App")
st.markdown("This application uses a KNN model to predict breast cancer. Early detection can significantly improve treatment outcomes. Provide tumor characteristics and click **Predict** to evaluate the likelihood of malignancy.")

st.info("⚠️ Note: This prediction is for educational purposes only. Please consult with a medical professional for an accurate diagnosis.")

# Sidebar for inputs
st.sidebar.header("Input Tumor Characteristics")
def get_slider(label):
    return st.sidebar.slider(label, 1, 10, 1)

features = [
    get_slider('Clump Thickness'),
    get_slider('Uniformity of Cell Size'),
    get_slider('Uniformity of Cell Shape'),
    get_slider('Marginal Adhesion'),
    get_slider('Single Epithelial Cell Size'),
    get_slider('Bare Nuclei'),
    get_slider('Bland Chromatin'),
    get_slider('Normal Nucleoli'),
    get_slider('Mitoses')
]

# Scale input data using a fresh StandardScaler
input_data_scaled = scaler.fit_transform([features])

# Prediction
if st.button("Predict 🔍"):
    try:
        prediction = knn_model.predict(input_data_scaled)
        probabilities = knn_model.predict_proba(input_data_scaled)
        benign_prob = probabilities[0][0] * 100
        malignant_prob = probabilities[0][1] * 100
        
        st.write(f"🔎 Prediction Confidence: **Benign:** {benign_prob:.2f}% | **Malignant:** {malignant_prob:.2f}%")
        
        if prediction[0] == 2:
            st.success("✅ The tumor is predicted to be **Benign**. Please continue regular check-ups and maintain a healthy lifestyle.")
        else:
            st.error(
                "💔 The tumor is predicted to be **Malignant**. While this result may be concerning, remember that early diagnosis significantly improves the chances of successful treatment. Please consult with an oncologist immediately for further evaluation and appropriate medical advice. You are not alone, and support is available."
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.write("~ Vaidehi")