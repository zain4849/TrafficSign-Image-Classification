import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About the Project")
st.markdown("""
## 🚦 Traffic Sign Classification using Deep Learning

This project uses deep learning to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

### 🧠 Model Architecture
- Custom CNN architecture
- Trained on 43 different traffic sign classes
- Achieved ~94% accuracy on test set

### 🛠️ Technologies Used
- TensorFlow/Keras
- Streamlit
- Python 3.12
- OpenCV

### 📊 Dataset
The GTSRB dataset contains more than 50,000 images of traffic signs in 43 classes.
""")