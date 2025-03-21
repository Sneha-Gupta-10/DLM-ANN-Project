import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import requests

# GitHub raw file link
MODEL_URL = "https://raw.githubusercontent.com/Sneha-Gupta-10/DLM-ANN-Project/refs/heads/main/sg47_ann_model.h5"
MODEL_PATH = "sg47_ann_model.h5"

@st.cache_resource
def load_sg47_model():
    # Download the model if it doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
sg47_model = load_sg47_model()


# Step 5: Streamlit Dashboard Code
def sg47_run_dashboard():
    st.title("Marketing Campaign Success Prediction")
    
    # Sidebar for hyperparameters
    epochs = st.sidebar.slider("Epochs", 10, 100, step=10, value=50)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, step=8, value=16)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, step=0.0001, format="%.4f", value=0.001)
    activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"], index=0)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, step=1, value=3)
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 8, 128, step=8, value=32)
    
    # Load the model inside the function
    sg47_model = load_sg47_model()

    # Now evaluate
    accuracy = sg47_model.evaluate(sg47_X_test, sg47_y_test, verbose=0)[1]

    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")
    
    # Visualizations
    st.subheader("Model Accuracy Over Epochs")
    fig, ax = plt.subplots()
    ax.plot(sg47_history.history['accuracy'], label='Train Accuracy')
    ax.plot(sg47_history.history['val_accuracy'], label='Validation Accuracy')
    ax.legend()
    ax.set_title("Model Accuracy")
    st.pyplot(fig)
    
    # New: Feature Importance Visualization (using mean absolute weights)
    st.subheader("Feature Importance")
    feature_importance = np.mean(np.abs(sg47_model.get_weights()[0]), axis=1)
    feature_names = sg47_X.columns
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax)
    ax.set_title("Feature Importance based on ANN Weights")
    st.pyplot(fig)
    
    # New: Distribution of Predicted Probabilities
    st.subheader("Distribution of Predicted Probabilities")
    sg47_y_prob = sg47_model.predict(sg47_X_test)
    fig, ax = plt.subplots()
    sns.histplot(sg47_y_prob, bins=20, kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability of Success")
    st.pyplot(fig)
    
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(sg47_y_test, (sg47_model.predict(sg47_X_test) > 0.5).astype("int32").ravel()), annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Run Streamlit
if __name__ == "__main__":
    sg47_run_dashboard()
