import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load dataset
DATASET_URL = "https://raw.githubusercontent.com/Sneha-Gupta-10/DLM-ANN-Project/main/digital_marketing_campaigns_smes_.csv"
df = pd.read_csv(DATASET_URL)

# Data Preprocessing
df.drop(columns=['campaign_id'], inplace=True)

categorical_cols = ['company_size', 'industry', 'marketing_channel', 'target_audience_area', 'target_audience_age',
                    'region', 'device', 'operating_system', 'browser', 'success']
ordinal_encoder = OrdinalEncoder()
df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])

numerical_cols = ['ad_spend', 'duration', 'engagement_metric', 'conversion_rate',
                  'budget_allocation', 'audience_reach', 'device_conversion_rate',
                  'os_conversion_rate', 'browser_conversion_rate']
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split data
X = df.drop(columns=['success'])
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5504714)

# Streamlit UI
st.title("Digital Marketing Campaign Success Prediction")

# Sidebar - Select Hyperparameters
epochs = st.sidebar.slider("Epochs", 10, 100, step=10, value=50)
batch_size = st.sidebar.slider("Batch Size", 8, 64, step=8, value=16)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, step=0.0001, format="%.4f", value=0.001)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], index=0)
num_layers = st.sidebar.slider("Number of Layers", 1, 5, step=1, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 8, 128, step=8, value=32)

# Train model function (cached for performance)
@st.cache_resource
def train_ann(epochs, batch_size, learning_rate, activation_function, num_layers, neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation_function, input_shape=(X_train.shape[1],)))
    
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    
    model.add(Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
    return model, history

# Show Model Summary Button (Result appears on Main Page)
if st.button("Show Model Summary"):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation_function, input_shape=(X_train.shape[1],)))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))
    st.text("\n".join(summary_str))  # Shows the model summary on the main page

# Train Model Button
train_model_clicked = st.button("Train Model")

if train_model_clicked:
    model, history = train_ann(epochs, batch_size, learning_rate, activation_function, num_layers, neurons_per_layer)
    
    # Show success message
    st.success("Model trained successfully!")
    
    # Compute evaluation metrics
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")
    
    # Visualizations

    # Accuracy Graph
    st.subheader("Model Accuracy Over Epochs")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax.legend()
    ax.set_title("Training vs Validation Accuracy")
    st.pyplot(fig)

    # Loss Graph
    st.subheader("Model Loss Over Epochs")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history['loss'], label='Train Loss', linestyle='dashed', color='blue')
    ax.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed', color='red')
    ax.legend()
    ax.set_title("Training vs Validation Loss")
    st.pyplot(fig)

    # Precision, Recall & F1-Score
    st.subheader("Precision, Recall & F1-Score")
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_df = pd.DataFrame({"Metric": ["Precision", "Recall", "F1-Score"], "Score": [precision, recall, f1]})
    sns.barplot(x="Metric", y="Score", data=metrics_df, ax=ax)
    ax.set_title("Precision, Recall & F1-Score")
    st.pyplot(fig)

    # Prediction Probability Distribution
    st.subheader("Prediction Probability Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(y_pred_prob, bins=20, kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability of Success")
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pd.crosstab(y_test, y_pred.ravel()), annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
