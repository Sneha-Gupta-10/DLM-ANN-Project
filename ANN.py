import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# File ID from Google Drive link
sg47_file_id = '1-mdmgqVhtp3fqMKuDK1yBzV6MH2OiXYj'

# Construct the download URL
sg47_download_url = f'https://drive.google.com/uc?id={sg47_file_id}'

# Load the dataset
sg47_DigitalMarketingCampaigns_data = pd.read_csv(sg47_download_url)
sg47_DigitalMarketingCampaigns_data

# Step 2: Data Preprocessing
sg47_DigitalMarketingCampaigns_data.drop(columns=['campaign_id'], inplace=True)  # Remove unnecessary ID column

# Encode categorical variables using Ordinal Encoding
sg47_categorical_cols = ['company_size', 'industry', 'marketing_channel', 'target_audience_area','target_audience_age',
                          'region', 'device', 'operating_system', 'browser','success']
sg47_ordinal_encoder = OrdinalEncoder()
sg47_DigitalMarketingCampaigns_data[sg47_categorical_cols] = sg47_ordinal_encoder.fit_transform(sg47_DigitalMarketingCampaigns_data[sg47_categorical_cols])
sg47_DigitalMarketingCampaigns_data

# Normalize numerical features using Min-Max Scaling
sg47_numerical_cols = ['ad_spend', 'duration', 'engagement_metric', 'conversion_rate',
                        'budget_allocation', 'audience_reach', 'device_conversion_rate',
                        'os_conversion_rate', 'browser_conversion_rate']
sg47_scaler = MinMaxScaler()
sg47_DigitalMarketingCampaigns_data[sg47_numerical_cols] = sg47_scaler.fit_transform(sg47_DigitalMarketingCampaigns_data[sg47_numerical_cols])
sg47_DigitalMarketingCampaigns_data

# Step 3: Split dataset into training and testing
sg47_X = sg47_DigitalMarketingCampaigns_data.drop(columns=['success'])
sg47_y = sg47_DigitalMarketingCampaigns_data['success']
sg47_X_train, sg47_X_test, sg47_y_train, sg47_y_test = train_test_split(sg47_X, sg47_y, test_size=0.2, random_state=5504714)

# Step 4: Build ANN Model
sg47_model = Sequential([
    Dense(64, activation='relu', input_shape=(sg47_X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

sg47_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train Model
sg47_history = sg47_model.fit(sg47_X_train, sg47_y_train, epochs=50, batch_size=16, validation_data=(sg47_X_test, sg47_y_test), verbose=1)

!pip install streamlit

import streamlit as st
print("Streamlit imported successfully!")

# Step 6: Streamlit Dashboard Code
def sg47_run_dashboard():
    st.title("Digital Marketing Campaign Success Prediction")
    
    # Sidebar for hyperparameters
    epochs = st.sidebar.slider("Epochs", 10, 100, step=10, value=50)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, step=8, value=16)
    
    # Model accuracy display
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
    
    st.subheader("Confusion Matrix")
    sg47_y_pred = (sg47_model.predict(sg47_X_test) > 0.5).astype("int32")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(sg47_y_test, sg47_y_pred.ravel()), annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # Prediction button
    if st.button("Predict on Test Data"):
        st.write("Predictions:", sg47_y_pred[:10])

# Run Streamlit
if __name__ == "__main__":
    sg47_run_dashboard()