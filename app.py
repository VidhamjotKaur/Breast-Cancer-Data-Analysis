import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the Streamlit Breast Cancer Prediction App.
    
    This function orchestrates the loading of the model, scaler, selected features,
    and dataset. It handles user input, preprocessing, prediction, and displays
    the results.
    """
    # App Title and Description
    st.title("Breast Cancer Prediction App")
    st.write("""
    This app predicts whether a breast tumor is **Malignant** or **Benign** based on various features.
    Adjust the parameters in the sidebar to make predictions.
    """)

    # Load necessary components
    model = load_model()
    scaler = load_scaler()
    selected_features = load_selected_features()
    X = load_data()

    # Get user input
    user_input = get_user_input(selected_features, X)

    # Preprocess user input
    user_input_scaled = preprocess_input(user_input, scaler)

    # Make prediction
    prediction, prediction_proba = make_prediction(model, user_input_scaled)

    # Display results
    display_results(prediction, prediction_proba)

@st.cache_resource
def load_model():
    """
    Load the pre-trained MLP model from a file.
    
    Returns:
        model: Loaded machine learning model.
    """
    try:
        model = joblib.load('mlp_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'mlp_model.pkl' not found.")
        st.stop()

@st.cache_resource
def load_scaler():
    """
    Load the StandardScaler object from a file.
    
    Returns:
        scaler: Loaded scaler object.
    """
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found.")
        st.stop()

@st.cache_resource
def load_selected_features():
    """
    Load the list of selected feature names from a file.
    
    Returns:
        selected_features (list): List of selected feature names.
    """
    try:
        selected_features = joblib.load('selected_features.pkl')
        return selected_features
    except FileNotFoundError:
        st.error("Selected features file 'selected_features.pkl' not found.")
        st.stop()

@st.cache_data
def load_data():
    """
    Load the breast cancer dataset features.
    
    Returns:
        X (pd.DataFrame): DataFrame containing feature data.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    return X

def get_user_input(selected_features, X):
    """
    Collect user input parameters through Streamlit sidebar sliders.
    
    Args:
        selected_features (list): List of selected feature names.
        X (pd.DataFrame): DataFrame containing feature data.
    
    Returns:
        features (pd.DataFrame): DataFrame containing user input features.
    """
    st.sidebar.header('User Input Parameters')

    user_data = {}
    for feature in selected_features:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        user_data[feature] = st.sidebar.slider(
            feature, min_val, max_val, mean_val
        )

    features = pd.DataFrame(user_data, index=[0])
    return features

def preprocess_input(user_input, scaler):
    """
    Preprocess the user input by scaling using the loaded scaler.
    
    Args:
        user_input (pd.DataFrame): DataFrame containing user input features.
        scaler: Loaded scaler object.
    
    Returns:
        user_input_scaled (np.ndarray): Scaled user input features.
    """
    try:
        user_input_scaled = scaler.transform(user_input)
        return user_input_scaled
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

def make_prediction(model, user_input_scaled):
    """
    Make a prediction using the pre-trained model on the scaled user input.
    
    Args:
        model: Loaded machine learning model.
        user_input_scaled (np.ndarray): Scaled user input features.
    
    Returns:
        prediction (np.ndarray): Predicted class label.
        prediction_proba (np.ndarray): Probability estimates for each class.
    """
    try:
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

def display_results(prediction, prediction_proba):
    """
    Display the prediction results and probability in the Streamlit app.
    
    Args:
        prediction (np.ndarray): Predicted class label.
        prediction_proba (np.ndarray): Probability estimates for each class.
    """
    st.subheader('Prediction')
    cancer_types = np.array(['Malignant', 'Benign'])
    st.write(cancer_types[prediction][0])

    st.subheader('Prediction Probability')
    fig, ax = plt.subplots()
    ax.bar(cancer_types, prediction_proba[0], color=['red', 'green'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    for i, v in enumerate(prediction_proba[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
