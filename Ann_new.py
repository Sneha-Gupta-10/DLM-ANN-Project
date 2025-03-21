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
import pickle

# GitHub file links
MODEL_URL = "https://raw.githubusercontent.com/Sneha-Gupta-10/DLM-ANN-Project/main/sg47_ann_model.h5"
DATASET_URL = "https://raw.githubusercontent.com/Sneha-Gupta-10/DLM-ANN-Project/main/digital_marketing_campaigns_smes_.csv"
HISTORY_URL = "https://raw.githubusercontent.com/Sneha-Gupta-10/DLM-ANN-Project/main/sg47_history.pkl"

MODEL_PATH = "sg47_ann_model.h5"
DATASET_PATH = "digital_marketing_campaigns_smes_.csv"
HISTORY_PATH = "sg47_history.pkl"

# Download model if not found
@st.cache_resource
def load_sg47_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

sg47_model = load_sg47_model()

# Download dataset if not found
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "wb") as f:
        response = requests.get(DATASET_URL)
        f.write(response.content)

# File ID from Google Drive link
sg47_file_id = '1-mdmgqVhtp3fqMKuDK1yBzV6MH2OiXYj'

# Construct the download URL
sg47_download_url = f'https://drive.google.com/uc?id={sg47_file_id}'

# Load the dataset
sg47_DigitalMarketingCampaigns_data = pd.read_csv(sg47_download_url)

# Step 2: Data Preprocessing
sg47_DigitalMarketingCampaigns_data.drop(columns=['campaign_id'], inplace=True)  # Remove unnecessary ID column

# Encode categorical variables using Ordinal Encoding
sg47_categorical_cols = ['company_size', 'industry', 'marketing_channel', 'target_audience_area','target_audience_age',
                          'region', 'device', 'operating_system', 'browser','success']
sg47_ordinal_encoder = OrdinalEncoder()
sg47_DigitalMarketingCampaigns_data[sg47_categorical_cols] = sg47_ordinal_encoder.fit_transform(sg47_DigitalMarketingCampaigns_data[sg47_categorical_cols])

# Normalize numerical features using Min-Max Scaling
sg47_numerical_cols = ['ad_spend', 'duration', 'engagement_metric', 'conversion_rate',
                        'budget_allocation', 'audience_reach', 'device_conversion_rate',
                        'os_conversion_rate', 'browser_conversion_rate']
sg47_scaler = MinMaxScaler()
sg47_DigitalMarketingCampaigns_data[sg47_numerical_cols] = sg47_scaler.fit_transform(sg47_DigitalMarketingCampaigns_data[sg47_numerical_cols])

# Step 3: Split dataset into training and testing
sg47_X = sg47_DigitalMarketingCampaigns_data.drop(columns=['success'])
sg47_y = sg47_DigitalMarketingCampaigns_data['success']
sg47_X_train, sg47_X_test, sg47_y_train, sg47_y_test = train_test_split(sg47_X, sg47_y, test_size=0.2, random_state=5504714)

# Download and load training history
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "wb") as f:
        response = requests.get(HISTORY_URL)
        f.write(response.content)

with open(HISTORY_PATH, "rb") as f:
    sg47_history = pickle.load(f)

# Streamlit Dashboard
def sg47_run_dashboard():
    st.title("Marketing Campaign Success Prediction")

    # Model Accuracy
    accuracy = sg47_model.evaluate(sg47_X_test, sg47_y_test, verbose=0)[1]
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

    # ADDING VISUALIZATIONS BELOW

    ## **Model Accuracy Over Epochs**
    st.subheader("Model Accuracy Over Epochs")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sg47_history['accuracy'], label='Train Accuracy', color='blue')
    ax.plot(sg47_history['val_accuracy'], label='Validation Accuracy', color='red')
    ax.legend()
    ax.set_title("Training vs Validation Accuracy")
    st.pyplot(fig)

    ## **Feature Importance Visualization**
    st.subheader("Feature Importance")
    try:
        feature_importance = np.mean(np.abs(sg47_model.get_weights()[0]), axis=1)
        feature_names = sg47_X.columns
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=feature_importance, y=feature_names, ax=ax)
        ax.set_title("Feature Importance based on ANN Weights")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature Importance could not be calculated: {e}")

    ## **Prediction Probability Distribution**
    st.subheader("Prediction Probability Distribution")
    sg47_y_prob = sg47_model.predict(sg47_X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(sg47_y_prob, bins=20, kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability of Success")
    st.pyplot(fig)

    ## **Confusion Matrix**
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pd.crosstab(sg47_y_test, (sg47_model.predict(sg47_X_test) > 0.5).astype("int32").ravel()), annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Run Streamlit
if __name__ == "__main__":
    sg47_run_dashboard()
