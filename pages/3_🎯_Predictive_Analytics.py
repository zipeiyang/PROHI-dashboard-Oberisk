import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import os
import time

# Set page configuration
st.set_page_config(page_title="Predictive Analytics", layout="wide")
st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

# Define constants
CATEGORY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

# Load pre-trained model
@st.cache_resource
def load_model():
    """Load pre-trained model from file"""
    try:
        # Update this path to where your model is saved
        model_path = "./jupyter-notebooks/models/obesity_risk_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("Model loaded successfully!")
            return model
        else:
            st.error(f"Model file not found at: {model_path}")
            st.info("Please ensure the model file exists or train the model first.")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# User input form - moved to main page
def get_user_input():
    """Create user input form in main page"""
    st.header("Input Your Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.slider("Age", 18, 80, 30)
        # height = st.slider("Height (m)", 1.4, 2.2, 1.7)
        # weight = st.slider("Weight (kg)", 40.0, 150.0, 70.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        
        st.subheader("Eating Habits")
        veggie_per_meal = st.slider("Vegetables per meal", 1.0, 3.0, 2.0)
        meals_daily = st.slider("Meals daily", 1.0, 4.0, 3.0)
        water_daily = st.slider("Water daily (liters)", 1.0, 3.0, 2.0)
        often_high_calorie_intake = st.selectbox("Often high calorie intake", ["yes", "no"])
        freq_snack = st.selectbox("Snack frequency", ["Sometimes", "Frequently", "Always", "no"])
    
    with col2:
        st.subheader("Lifestyle Factors")
        physical_activity = st.slider("Physical activity (days/week)", 0.0, 3.0, 1.0)
        technological_devices = st.slider("Technology devices usage (hours/day)", 0.0, 2.0, 1.0)
        family_history = st.selectbox("Family history of overweight", ["yes", "no"])
        smoking = st.selectbox("Smoking", ["yes", "no"])
        monitor_calorie = st.selectbox("Monitor calorie intake", ["yes", "no"])
        freq_alcohol = st.selectbox("Alcohol frequency", ["Sometimes", "Frequently", "Always", "no"])
        transport = st.selectbox("Transportation method", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    
    # Create a DataFrame from user input
    user_data = {
        'age': age,
        # 'height': height,
        # 'weight': weight,
        'veggie_per_meal': veggie_per_meal,
        'meals_daily': meals_daily,
        'water_daily': water_daily,
        'physical_activity': physical_activity,
        'technological_devices': technological_devices,
        'sex': sex,
        'family_history': family_history,
        'often_high_calorie_intake': often_high_calorie_intake,
        'freq_snack': freq_snack,
        'smoking': smoking,
        'monitor_calorie': monitor_calorie,
        'freq_alcohol': freq_alcohol,
        'transport': transport
    }
    
    return pd.DataFrame(user_data, index=[0])

# Display prediction results
def display_prediction_result(prediction, prediction_proba):
    """Display prediction results with appropriate styling"""
    if prediction[0] == 1:
        st.error(f"**High Obesity Risk** (Probability: {prediction_proba[0][1]:.2%})")
        st.markdown("""
        **Recommendations:**
        - Consult with a healthcare professional
        - Consider dietary changes and increased physical activity
        - Monitor your weight regularly
        - Reduce consumption of high-calorie foods
        - Increase daily physical activity to at least 30 minutes
        """)
    else:
        st.success(f"**Low Obesity Risk** (Probability: {prediction_proba[0][0]:.2%})")
        st.markdown("""
        **Recommendations:**
        - Maintain your current healthy habits
        - Continue regular physical activity
        - Monitor your weight to maintain healthy levels
        - Ensure balanced nutrition in your diet
        - Stay hydrated and get adequate sleep
        """)

# Main application
def main():
    st.header("Obesity Risk Prediction")
    st.markdown("""
    The Oberisk tool predicts your risk of obesity based on your lifestyle, demographic factors (age/sex), and family-history of obesity.
    Please enter your information below and click the 'Predict Risk' button to get your personalized obesity risk assessment.
    """)
    
    # Load the pre-trained model
    with st.spinner("Loading prediction model..."):
        model = load_model()
        time.sleep(1)  # Simulate loading time
    
    if model is None:
        st.error("Cannot proceed without a trained model. Please ensure the model file exists.")
        return
    
    # Get user input
    user_input = get_user_input()
    
    # Display user input
    st.subheader("Your Input Summary")
    st.dataframe(user_input)
    
    # Prediction button
    if st.button("Predict Obesity Risk", type="primary"):
        with st.spinner("Analyzing your information..."):
            # Simulate processing time for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing... {i+1}%")
                time.sleep(0.02)  # Simulate processing time
            
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            
            # Make prediction
            try:
                prediction = model.predict(user_input)
                prediction_proba = model.predict_proba(user_input)
                
                # Display results
                st.subheader("Prediction Result")
                display_prediction_result(prediction, prediction_proba)
                
                # Additional information
                with st.expander("View detailed analysis"):
                    st.write("Prediction probabilities:")
                    proba_df = pd.DataFrame({
                        'Class': ['No Risk', 'At Risk'],
                        'Probability': prediction_proba[0]
                    })
                    st.dataframe(proba_df)
                    
                    # Simple visualization
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(['No Risk', 'At Risk'], prediction_proba[0], color=['green', 'red'])
                    ax.set_xlabel('Probability')
                    ax.set_title('Risk Probability Distribution')
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        # Reset progress bar
        progress_bar.empty()
        status_text.empty()

    # Add some information about the model
    st.markdown("---")
    st.subheader("Information about the Prediction Model")
    st.markdown("""
    This prediction model was trained on a comprehensive dataset of lifestyle and health factors.
    It uses advanced machine learning techniques to assess an individual's obesity risk based on the following factors:
    - Demographic information (age, sex)
    - Eating habits (meal frequency, vegetable intake, high-calorie food consumption)
    - Lifestyle factors (physical activity, technology usage, transportation methods)
    - Family history and health behaviors
    
    **Note:** This tool is for informational purposes only and should only be used as a tool to help you understand how your lifestyle choices impact obesity risk. It DOES NOT replace professional medical advice. Please seek immediate medical attention and contact your healthcare provider if you are feeling unwell.
     Your data will not be stored for any purposes. Please feel free to contact the team at Oberisk or our main sponsor Region Stockholm if you have any questions about this tool or would like to receive more information about it.       
    """)

if __name__ == "__main__":
    main()