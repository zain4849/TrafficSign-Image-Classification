import streamlit as st
import plotly.express as px
import numpy as np
import os
from utils.model_utils import SIGN_CLASSES

st.set_page_config(page_title="Dataset Info", page_icon="📊", layout="wide")

@st.cache_data
def load_training_data():
    try:
        X_train = np.load('./data/X_train.npy')
        y_train = np.load('./data/y_train.npy')
        return X_train, y_train
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return None, None

X_train, y_train = load_training_data()

st.title("Dataset Information")

st.markdown("""
## German Traffic Sign Recognition Benchmark (GTSRB)

The dataset consists of:
- 43 different traffic sign classes
- Over 50,000 images in total
- Real-world images with varying lighting and weather conditions
""")

# Class distribution display
if X_train is not None and y_train is not None:
    st.subheader("Class Distribution")
    true_classes = np.argmax(y_train, axis=1)  # One-hot to class indices
    class_counts = [np.sum(true_classes == i) for i in range(43)]
    
    fig = px.bar(
        x=list(SIGN_CLASSES.values()),
        y=class_counts,
        title='Number of Images per Class',
        template='plotly_dark'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Traffic Sign Class",
        yaxis_title="Number of Images",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sample images display
    st.subheader("Sample Images")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        idx = np.random.randint(0, len(X_train))
        col.image(
            X_train[idx], 
            caption=SIGN_CLASSES[np.argmax(y_train[idx])],
            use_container_width=True
        )
else:
    st.warning("Please ensure training data files (X_train.npy and y_train.npy) are present in the data directory.")