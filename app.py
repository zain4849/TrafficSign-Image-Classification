import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from utils.model_utils import load_model, predict_sign
from utils.preprocessing import preprocess_image

st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 Traffic Sign Classification")
st.markdown("""
### Upload a traffic sign image for classification
This model can identify 43 different types of traffic signs.
""")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        with st.spinner('Analyzing...'):
            try:
                processed_image = preprocess_image(image)
                model = load_model('./models/custom_cnn_best.keras')
                
                dummy_input = tf.zeros((1, 64, 64, 3))
                model(dummy_input)
                
                prediction, confidence = predict_sign(model, processed_image)
                
                st.success("#### Prediction Results")
                st.markdown(f"**Sign Type:** {prediction}")
                st.progress(float(confidence))
                st.info(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")