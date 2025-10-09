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
st.set_page_config(page_title="Obesity Risk Factors", layout="wide")
st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

# Define constants
CATEGORY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

# Initialize session state with proper structure
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "counterfactual_values" not in st.session_state:
    st.session_state.counterfactual_values = {}
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if "current_user_input" not in st.session_state:
    st.session_state.current_user_input = None

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
    st.subheader("Input Your Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.slider("Age", 18, 80, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        family_history = st.selectbox("Family history of overweight", ["yes", "no"])
        
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
        smoking = st.selectbox("Smoking", ["yes", "no"])
        monitor_calorie = st.selectbox("Monitor calorie intake", ["yes", "no"])
        freq_alcohol = st.selectbox("Alcohol frequency", ["Sometimes", "Frequently", "Always", "no"])
        transport = st.selectbox("Transportation method", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    
    # Create a DataFrame from user input
    user_data = {
        'age': age,
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

def initialize_counterfactual_values(user_input):
    """Initialize counterfactual values from user input"""
    return {
        'physical_activity': float(user_input["physical_activity"].iloc[0]),
        'technological_devices': float(user_input["technological_devices"].iloc[0]),
        'meals_daily': float(user_input["meals_daily"].iloc[0]),
        'veggie_per_meal': float(user_input["veggie_per_meal"].iloc[0]),
        'water_daily': float(user_input["water_daily"].iloc[0]),
        'freq_snack': user_input["freq_snack"].iloc[0],
        'freq_alcohol': user_input["freq_alcohol"].iloc[0],
        'monitor_calorie': user_input["monitor_calorie"].iloc[0],
        'often_high_calorie_intake': user_input["often_high_calorie_intake"].iloc[0],
        'smoking': user_input["smoking"].iloc[0],
        'transport': user_input["transport"].iloc[0]
    }

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
    
    # Store current user input in session state
    st.session_state.current_user_input = user_input
    
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
                time.sleep(0.02)
            
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            
            # Make prediction
            try:
                prediction = model.predict(user_input)
                prediction_proba = model.predict_proba(user_input)
                
                # Store everything in session state
                st.session_state.user_input = user_input
                st.session_state.prediction = prediction
                st.session_state.prediction_proba = prediction_proba
                st.session_state.prediction_made = True
                
                # Initialize counterfactual values with original values
                st.session_state.counterfactual_values = initialize_counterfactual_values(user_input)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        # Reset progress bar
        progress_bar.empty()
        status_text.empty()

    # Display prediction results if available
    if st.session_state.prediction_made:
        st.subheader("Prediction Result")
        display_prediction_result(
            st.session_state.prediction, 
            st.session_state.prediction_proba
        )

        # Additional information
        with st.expander("View detailed analysis"):
            st.write("Prediction probabilities:")
            proba_df = pd.DataFrame({
                'Class': ['No Risk', 'At Risk'],
                'Probability': st.session_state.prediction_proba[0]
            })
            st.dataframe(proba_df)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.barh(['No Risk', 'At Risk'], st.session_state.prediction_proba[0], color=['green', 'red'])
                ax.set_xlabel('Probability')
                ax.set_title('Risk Probability Distribution')
                plt.tight_layout()
                st.pyplot(fig)
    
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
     Your data will not be stored for any purposes. Please feel free to contact the team at Oberisk or Region Stockholm if you have any questions about this tool or would like to receive more information about it.       
    """)

    # Counterfactual Analysis
    if st.session_state.prediction_made:
        st.markdown("---")
        st.subheader("If You Do Some Changes now...")
        st.markdown("""
        Adjust the sliders below to explore **how changing your habits might affect your predicted obesity risk**.  
        This interactive tool lets you simulate different lifestyle scenarios and instantly see the updated prediction.
        """)

        # Get current counterfactual values from session state
        cf_values = st.session_state.counterfactual_values
        
        # Use reset counter to force widget updates
        reset_suffix = f"_reset_{st.session_state.reset_counter}"
        
        col1, col2 = st.columns(2)

        with col1:
            # Update session state values when sliders change
            cf_values['physical_activity'] = st.slider(
                "Physical activity (days/week)",
                0.0, 3.0, cf_values['physical_activity'], step=0.5, key=f"cf_physical_activity{reset_suffix}"
            )
            cf_values['technological_devices'] = st.slider(
                "Technology use (hours/day)",
                0.0, 2.0, cf_values['technological_devices'], step=0.5, key=f"cf_technological_devices{reset_suffix}"
            )
            cf_values['meals_daily'] = st.slider(
                "Meals per day", 1.0, 4.0, cf_values['meals_daily'], step=0.5, key=f"cf_meals_daily{reset_suffix}"
            )
            cf_values['veggie_per_meal'] = st.slider(
                "Vegetables per meal", 1.0, 3.0, cf_values['veggie_per_meal'], step=0.5, key=f"cf_veggie_per_meal{reset_suffix}"
            )
            cf_values['water_daily'] = st.slider(
                "Water intake (liters/day)",
                1.0, 3.0, cf_values['water_daily'], step=0.5, key=f"cf_water_daily{reset_suffix}"
            )

        with col2:
            cf_values['freq_snack'] = st.selectbox(
                "Snack frequency", ["no", "Sometimes", "Frequently", "Always"],
                index=["no", "Sometimes", "Frequently", "Always"].index(cf_values['freq_snack']),
                key=f"cf_freq_snack{reset_suffix}"
            )
            cf_values['freq_alcohol'] = st.selectbox(
                "Alcohol frequency", ["no", "Sometimes", "Frequently", "Always"],
                index=["no", "Sometimes", "Frequently", "Always"].index(cf_values['freq_alcohol']),
                key=f"cf_freq_alcohol{reset_suffix}"
            )
            cf_values['monitor_calorie'] = st.selectbox(
                "Monitor calorie intake", ["yes", "no"],
                index=["yes", "no"].index(cf_values['monitor_calorie']),
                key=f"cf_monitor_calorie{reset_suffix}"
            )
            cf_values['often_high_calorie_intake'] = st.selectbox(
                "Often high calorie intake", ["yes", "no"],
                index=["yes", "no"].index(cf_values['often_high_calorie_intake']),
                key=f"cf_often_high_calorie_intake{reset_suffix}"
            )
            cf_values['smoking'] = st.selectbox(
                "Smoking", ["yes", "no"],
                index=["yes", "no"].index(cf_values['smoking']),
                key=f"cf_smoking{reset_suffix}"
            )
            cf_values['transport'] = st.selectbox(
                "Transportation method", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
                index=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"].index(cf_values['transport']),
                key=f"cf_transport{reset_suffix}"
            )

        # Update session state
        st.session_state.counterfactual_values = cf_values

        # Create counterfactual input DataFrame
        cf_input = st.session_state.user_input.copy()
        for key, value in cf_values.items():
            cf_input[key] = value

        # Predict again using modified inputs
        try:
            cf_pred = model.predict(cf_input)
            cf_proba = model.predict_proba(cf_input)[0]

            st.markdown("### Updated Prediction with New Parameters")
            st.write(f"**New Predicted Risk:** {'High' if cf_pred[0]==1 else 'Low'} "
                    f"({cf_proba[1]*100:.2f}% chance of obesity risk)")

            delta = cf_proba[1] - st.session_state.prediction_proba[0][1]
            if delta > 0.01:  # Add threshold to avoid tiny floating point changes
                st.error(f"⚠️ Risk increased by {delta*100:.1f}% after changes.")
            elif delta < -0.01:
                st.success(f"✅ Risk decreased by {abs(delta)*100:.1f}% after changes.")
            else:
                st.info("No significant change in predicted risk.")

            # Reset button
            if st.button("Reset to Original", key=f"reset_button{reset_suffix}"):
                st.session_state.counterfactual_values = initialize_counterfactual_values(st.session_state.user_input)
                st.session_state.reset_counter += 1
                st.rerun()

            # Visualization
            st.markdown("**Comparison of Original vs Adjusted Prediction**")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))
                bars = ax.barh(["Original", "Adjusted"], 
                              [st.session_state.prediction_proba[0][1], cf_proba[1]], 
                              color=["gray", "orange"])
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability of High Obesity Risk")
                ax.bar_label(bars, fmt="%.2f", padding=3)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during counterfactual analysis: {e}")
    else:
        st.markdown("---")
        st.subheader("If You Do Some Changes now...")
        st.warning("Please run a prediction first to use the counterfactual analysis feature.")

if __name__ == "__main__":
    main()