import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import tensorflow as tf
from utils.model_utils import SIGN_CLASSES, load_model
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Model Analysis", page_icon="📊", layout="wide")

# Loading training histories
@st.cache_data
def load_training_history():
    histories = {}
    model_names = ['custom_cnn', 'mobilenetv2', 'resnet50']
    
    for name in model_names:
        history_path = f'./histories/{name}_history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[name] = json.load(f)
    return histories

# Loading test data and get predictions
@st.cache_data
def load_test_data():
    try:
        X_test = np.load('./data/X_test.npy')
        y_test = np.load('./data/y_test.npy')
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None, None

# Preprocessing functions for different models
def preprocess_for_mobilenet(images):
    """Scale images to [-1,1] range"""
    return images * 2 - 1

def preprocess_for_resnet(images):
    """ResNet50 specific preprocessing"""
    return tf.keras.applications.resnet50.preprocess_input(images * 255.0)

# Loading models and evaluate with proper preprocessing
@st.cache_data
def evaluate_models():
    X_test, y_test = load_test_data()
    if X_test is None:
        return None
    
    results = {}
    models = {
        'Custom CNN': ('./models/custom_cnn_best.keras', lambda x: x),  # No preprocessing needed
        'MobileNetV2': ('./models/mobilenetv2_best.keras', preprocess_for_mobilenet),
        'ResNet50': ('./models/resnet50_best.keras', preprocess_for_resnet)
    }
    
    for name, (path, preprocess_fn) in models.items():
        if os.path.exists(path):
            model = load_model(path)
            # Apply appropriate preprocessing
            X_test_processed = preprocess_fn(X_test)
            loss, acc = model.evaluate(X_test_processed, y_test, verbose=0)
            pred = model.predict(X_test_processed)
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            cm = confusion_matrix(true_classes, pred_classes)
            results[name] = {'accuracy': acc, 'loss': loss, 'cm': cm}
    
    return results

# Load data
histories = load_training_history()
model_results = evaluate_models()

st.title("📊 Model Performance Analysis")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Training History", "Model Performance", "Confusion Matrix", "Classification Report"])

with tab1:
    st.header("Training History")
    if histories:
        # Plot training history for each model separately
        for name, history in histories.items():
            model_name = name.replace('_', ' ').title()
            
            # Create two columns for accuracy and loss plots
            col1, col2 = st.columns(2)
            
            # Accuracy plot
            with col1:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history['accuracy'],
                    name='Train',
                    mode='lines',
                    line=dict(color='#4ECDC4')
                ))
                fig_acc.add_trace(go.Scatter(
                    y=history['val_accuracy'],
                    name='Validation',
                    mode='lines',
                    line=dict(color='#FF6B6B', dash='dash')
                ))
                fig_acc.update_layout(
                    title=f'{model_name} - Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # Loss plot
            with col2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history['loss'],
                    name='Train',
                    mode='lines',
                    line=dict(color='#4ECDC4')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history['val_loss'],
                    name='Validation',
                    mode='lines',
                    line=dict(color='#FF6B6B', dash='dash')
                ))
                fig_loss.update_layout(
                    title=f'{model_name} - Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Divider between models
            st.markdown("---")
    else:
        st.warning("No training history data available")

with tab2:
    st.header("Model Performance")
    if model_results:
        cols = st.columns(len(model_results))
        for i, (name, results) in enumerate(model_results.items()):
            with cols[i]:
                st.subheader(name)
                st.metric("Test Accuracy", f"{results['accuracy']:.2%}")
                st.metric("Test Loss", f"{results['loss']:.4f}")
    else:
        st.warning("No model evaluation results available")

with tab3:
    st.header("Confusion Matrix")
    if model_results:
        # Model selector for confusion matrix
        model_name = st.selectbox("Select Model", list(model_results.keys()))
        cm = model_results[model_name]['cm']
        
        # Normalise confusion matrix
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = px.imshow(
            cm_percent,
            labels=dict(x="Predicted", y="True"),
            x=[SIGN_CLASSES[i] for i in range(43)],
            y=[SIGN_CLASSES[i] for i in range(43)],
            title=f'Confusion Matrix - {model_name}',
            template='plotly_dark',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=800,
            width=1000
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No confusion matrix data available")

with tab4:
    st.header("Classification Report")
    if model_results:
        model_name = st.selectbox("Select Model", list(model_results.keys()), key="report_model_select")
        
        X_test, y_test = load_test_data()
        if X_test is not None:
            # Get model and its preprocessing function
            model_path = f'./models/{model_name.lower().replace(" ", "_")}_best.keras'
            preprocess_fn = {
                'Custom CNN': lambda x: x,
                'MobileNetV2': preprocess_for_mobilenet,
                'ResNet50': preprocess_for_resnet
            }[model_name]
            
            model = load_model(model_path)
            X_test_processed = preprocess_fn(X_test)
            y_pred = model.predict(X_test_processed)
            
            # Getting class predictions
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Generating classification report
            report = classification_report(
                y_true_classes,
                y_pred_classes,
                target_names=[SIGN_CLASSES[i] for i in range(43)],
                output_dict=True
            )
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                with st.expander("Detailed Metrics by Class", expanded=True):
                    data = []
                    for class_name in SIGN_CLASSES.values():
                        if class_name in report:
                            metrics = report[class_name]
                            data.append({
                                "Class": class_name,
                                "Precision": metrics['precision'],
                                "Recall": metrics['recall'],
                                "F1-Score": metrics['f1-score'],
                                "Support": int(metrics['support'])
                            })
                    
                    import pandas as pd
                    df = pd.DataFrame(data)
                    
                    # Background gradient to numeric columns
                    styled_df = df.style.background_gradient(
                        subset=['Precision', 'Recall', 'F1-Score'],
                        cmap='RdYlGn'
                    ).format({
                        'Precision': '{:.2%}',
                        'Recall': '{:.2%}',
                        'F1-Score': '{:.2%}',
                        'Support': '{:.0f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)
            
            with col2:
                # Overall metrics
                st.write("### Overall Performance")
                
                # Overall accuracy
                accuracy = report['accuracy']
                st.metric(
                    "Overall Accuracy",
                    f"{accuracy:.2%}",
                    delta=f"{accuracy-0.5:.2%} vs random"
                )
                
                # Macro averages
                st.metric(
                    "Macro Avg F1-Score",
                    f"{report['macro avg']['f1-score']:.2%}"
                )
                
                # Weighted averages
                st.metric(
                    "Weighted Avg F1-Score",
                    f"{report['weighted avg']['f1-score']:.2%}"
                )
                
                # F1-score distribution plot
                f1_scores = [report[c]['f1-score'] for c in SIGN_CLASSES.values() if c in report]
                fig = px.histogram(
                    f1_scores,
                    title="Distribution of F1-Scores",
                    labels={'value': 'F1-Score', 'count': 'Number of Classes'},
                    template='plotly_dark'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model evaluation results available")