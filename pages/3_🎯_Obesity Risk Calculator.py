import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import os
import time
import textwrap
from html import escape

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
    
    
# --- Helper: Likert buttons (1–3) with legend + built-in help tooltip ---
def likert_buttons(
    label: str,
    options: list[int],
    captions: list[str],
    key: str,
    columns: int = 4,
    help_text: str | None = None,
    default_index: int = 0,
) -> int:
    if key not in st.session_state:
        st.session_state[key] = options[default_index]

    # bold or normal
    st.markdown(f"{label}")

    cols = st.columns(columns)
    for i, opt in enumerate(options):
        btn_type = "primary" if st.session_state[key] == opt else "secondary"
        if cols[i % columns].button(
            str(opt),
            key=f"{key}_btn_{opt}",
            type=btn_type,
            use_container_width=True,
            help=help_text,
        ):
            st.session_state[key] = opt

    # NEW: legend as numbered list on separate lines
    legend_md = "\n".join([f"{o}. {c}" for o, c in zip(options, captions)])
    st.markdown(legend_md)

    return int(st.session_state[key])

def get_user_input():
    # ---------------- Row 1: 3 columns (Age | Sex | Family history) ----------------
    st.subheader("Basic Information")
    c1, c2, c3 = st.columns(3, gap="medium", vertical_alignment="top")

    with c1:
        st.markdown('<div class="orange-card">', unsafe_allow_html=True)
        age = st.number_input(
            "Age (years)", min_value=18, max_value=80, value=30, step=1,
            help="Adults only (18–80)."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="orange-card">', unsafe_allow_html=True)
        sex = st.selectbox(
            "Sex", ["Male", "Female"],
            help="Select the option that matches your sex at registration."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="orange-card">', unsafe_allow_html=True)
        family_history = st.selectbox(
            "Family history of overweight", ["yes", "no"],
            help="Did your close family members have overweight/obesity?"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Row 2: 2 columns (Eating Habits | Lifestyle Factors) ----------------
    left, right = st.columns(2, gap="large", vertical_alignment="top", border=True)

    with left:
        st.subheader("Eating Habits")

        veggie_per_meal = likert_buttons(
            label="Vegetables in meals",
            options=[1, 2, 3],
            captions=["Never", "Sometimes", "Always"],
            key="veggie_per_meal",
            help_text="Do you usually eat vegetables in your meals?"
        )

        meals_daily = likert_buttons(
            label="Main meals per day",
            options=[1, 2, 3],
            captions=["1–2 times", "2 times", "More than 3 times"],
            key="meals_daily",
            help_text="How many main meals do you have daily?"
        )

        water_daily = likert_buttons(
            label="Water intake per day",
            options=[1, 2, 3],
            captions=["Less than 1 L", "1–2 L", "More than 2 L"],
            key="water_daily",
            help_text="How much water do you drink daily?"
        )

        often_high_calorie_intake = st.selectbox(
            "Often eat high-calorie foods?",
            ["yes", "no"],
            help="Do you eat high-calorie food frequently (calories exceeding your estimated daily needs)?"
        )

        freq_snack = st.selectbox(
            "Snack frequency",
            ["no", "Sometimes", "Frequently", "Always"],
            help="How often do you eat snacks between main meals?"
        )

    with right:
        st.subheader("Lifestyle Factors")

        # If your model expects FAF 0–3, keep 0–3 here
        physical_activity_likert = likert_buttons(
            label="Physical activity (per week)",
            options=[0, 1, 2, 3],
            captions=["Do not have", "1–2 days", "2–4 days", "More than 5 days"],
            key="physical_activity",
            help_text="How often do you have physical activity?"
        )

        # If your model expects TUE 0–2, keep 0–2 here
        technological_devices_likert = likert_buttons(
            label="Technology use per day",
            options=[0, 1, 2],
            captions=["1–2 hours", "3–5 hours", "More than 5 hours"],
            key="technological_devices",
            help_text="Time using technological devices (cell phone, videogames, TV, computer)."
        )

        smoking = st.selectbox("Smoking", ["yes", "no"], help="Do you currently smoke?")
        monitor_calorie = st.selectbox("Monitor calorie intake", ["yes", "no"], help="Do you track or count calories?")
        freq_alcohol = st.selectbox(
            "Alcohol frequency", ["no", "Sometimes", "Frequently", "Always"],
            help="How often do you drink alcohol?"
        )
        transport = st.selectbox(
            "Usual transport mode",
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
            help="Your most common way of getting around."
        )
    # --- build user_data FIRST (values sent to the model) ---
    user_data = {
        "age": int(age),
        "veggie_per_meal": int(veggie_per_meal),
        "meals_daily": int(meals_daily),
        "water_daily": int(water_daily),
        "physical_activity": int(physical_activity_likert),        # FAF 0–3
        "technological_devices": int(technological_devices_likert),# TUE 0–2
        "sex": sex,
        "family_history": family_history,
        "often_high_calorie_intake": often_high_calorie_intake,
        "freq_snack": freq_snack,
        "smoking": smoking,
        "monitor_calorie": monitor_calorie,
        "freq_alcohol": freq_alcohol,
        "transport": transport,
    }

    # ---------------- Compact Quick Review (7 columns, short & tidy) ----------------
    st.markdown("---")
    st.subheader("Your input summary")

    css = textwrap.dedent("""
    <style>
    .kv-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 20px;}
    .kv-item { border: 1px solid #FF8C00; border-radius: 10px; padding: 10px 12px; background: #fff; }
    .kv-label { font-size: 12px; color: #6b7280; margin-bottom: 6px; line-height: 1.1; }
    .kv-value { font-weight: 600; font-size: 14px; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    @media (max-width: 1200px) { .kv-grid { grid-template-columns: repeat(4, 1fr); } }
    </style>
    """)
    st.markdown(css, unsafe_allow_html=True)

    ud = user_data  # read-only alias for display
    veggie_map = {1:"Never",2:"Sometimes",3:"Always"}
    meals_map  = {1:"1–2 times",2:"2 times",3:"More than 3 times"}
    water_map  = {1:"<1 L",2:"1–2 L",3:">2 L"}
    faf_map    = {0:"None",1:"1–2 days",2:"2–4 days",3:">5 days"}
    tue_map    = {0:"1–2 h",1:"3–5 h",2:">5 h"}

    pairs = [
        ("Age", f"{ud['age']}"),
        ("Sex", ud["sex"]),
        ("Family history", ud["family_history"]),
        ("Vegetables", veggie_map.get(ud["veggie_per_meal"], ud["veggie_per_meal"])),
        ("Meals/day", meals_map.get(ud["meals_daily"], ud["meals_daily"])),
        ("Water/day", water_map.get(ud["water_daily"], ud["water_daily"])),
        ("Physical activity", faf_map.get(ud["physical_activity"], ud["physical_activity"])),
        ("Tech use/day", tue_map.get(ud["technological_devices"], ud["technological_devices"])),
        ("Snacks", ud["freq_snack"]),
        ("High-cal foods", ud["often_high_calorie_intake"]),
        ("Smoking", ud["smoking"]),
        ("Monitor calories", ud["monitor_calorie"]),
        ("Alcohol", ud["freq_alcohol"]),
        ("Transport", ud["transport"]),
    ]

    cells = []
    for label, value in pairs:
        lbl = escape(str(label))
        val = escape(str(value))
        cells.append(f'<div class="kv-item"><div class="kv-label">{lbl}</div><div class="kv-value">{val}</div></div>')

    while len(cells) % 7 != 0:
        cells.append('<div class="kv-item" style="visibility:hidden;"></div>')

    html = '<div class="kv-grid">' + ''.join(cells) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

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
    st.header("Predict your obesity risk here!")
    st.markdown("""
    On this page, OBERISK will assists you to predicts your future risk of having obesity based on your age, sex, family history of obesity, eating habits, and lifestyle.
    Please **enter your information below** and click the **'Predict Obesity Risk'** button to get your personalized obesity risk assessment.
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
    st.badge("Your answers help estimate obesity risk. This is not a diagnosis", icon=":material/info:", color="red")

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
        st.subheader("What would happen if you changed your habits now?")
        st.markdown("""
        Try different choices below and see **how changing your habits might affect your predicted obesity risk**.  
        This interactive tool lets you simulate different lifestyle scenarios and instantly see the updated prediction.
        """)   
    # Compact card style (safe to inject again)
        st.markdown("""
        <style>
        .scroll-col { max-height: 480px; overflow-y: auto; padding-right: 6px; }
        .cf-grid-gap { margin-bottom: 10px; }
        </style>
        """, unsafe_allow_html=True)

        # Get current counterfactual values from session state
        cf_values = st.session_state.counterfactual_values
        cf_values["physical_activity"]     = int(round(float(cf_values.get("physical_activity", 0))))
        cf_values["technological_devices"] = int(round(float(cf_values.get("technological_devices", 0))))
        cf_values["meals_daily"]           = int(round(float(cf_values.get("meals_daily", 2))))
        cf_values["veggie_per_meal"]       = int(round(float(cf_values.get("veggie_per_meal", 2))))
        cf_values["water_daily"]           = int(round(float(cf_values.get("water_daily", 2))))
        
        # Use reset counter to force widget updates
        reset_suffix = f"_reset_{st.session_state.reset_counter}"
        
        col1, col2 = st.columns(2, gap ="medium")

        with col1:
            # Scrollable, bordered Streamlit container
            with st.container(height=460, border=True):
                st.subheader("Eating Habits", anchor=False)
                cf_values['veggie_per_meal'] = likert_buttons(
                    label="Vegetables in meals",
                    options=[1, 2, 3],
                    captions=["Never", "Sometimes", "Always"],
                    key=f"cf_veggie_per_meal{reset_suffix}",
                    help_text="Do you usually eat vegetables in your meals?"
                )

                cf_values['meals_daily'] = likert_buttons(
                    label="Main meals per day",
                    options=[1, 2, 3],
                    captions=["1–2 times", "2 times", "More than 3 times"],
                    key=f"cf_meals_daily{reset_suffix}",
                    help_text="How many main meals do you have daily?"
                )

                cf_values['water_daily'] = likert_buttons(
                    label="Water intake per day",
                    options=[1, 2, 3],
                    captions=["Less than 1 L", "1–2 L", "More than 2 L"],
                    key=f"cf_water_daily{reset_suffix}",
                    help_text="How much water do you drink daily?"
                )

                cf_values['often_high_calorie_intake'] = st.selectbox(
                    "Often eat high-calorie foods?", ["yes", "no"],
                    index=["yes","no"].index(cf_values.get('often_high_calorie_intake', 'no')),
                    help="Do you eat high-calorie food frequently (calories exceeding your daily needs)?",
                    key=f"cf_often_high_calorie_intake{reset_suffix}",
                )

                cf_values['freq_snack'] = st.selectbox(
                    "Snack frequency", ["no", "Sometimes", "Frequently", "Always"],
                    index=["no", "Sometimes", "Frequently", "Always"].index(cf_values.get('freq_snack', 'no')),
                    help="How often do you eat snacks between main meals?",
                    key=f"cf_freq_snack{reset_suffix}",
                )

        with col2:
            # Scrollable, bordered Streamlit container
            with st.container(height=460, border=True):
                st.subheader("Lifestyle Factors", anchor=False)

                cf_values['physical_activity'] = likert_buttons(
                    label="Physical activity (per week)",
                    options=[0, 1, 2, 3],
                    captions=["Do not have", "1–2 days", "2–4 days", "More than 5 days"],
                    key=f"cf_physical_activity{reset_suffix}",
                    help_text="How often do you have physical activity?"
                )

                cf_values['technological_devices'] = likert_buttons(
                    label="Technology use per day",
                    options=[0, 1, 2],
                    captions=["1–2 hours", "3–5 hours", "More than 5 hours"],
                    key=f"cf_technological_devices{reset_suffix}",
                    help_text="Time using technological devices (phone, games, TV, computer)."
                )

                cf_values['smoking'] = st.selectbox(
                    "Smoking", ["yes", "no"],
                    index=["yes","no"].index(cf_values.get('smoking', 'no')),
                    help="Do you currently smoke?",
                    key=f"cf_smoking{reset_suffix}",
                )

                cf_values['monitor_calorie'] = st.selectbox(
                    "Monitor calorie intake", ["yes", "no"],
                    index=["yes","no"].index(cf_values.get('monitor_calorie', 'no')),
                    help="Do you track or count calories?",
                    key=f"cf_monitor_calorie{reset_suffix}",
                )

                cf_values['freq_alcohol'] = st.selectbox(
                    "Alcohol frequency", ["no", "Sometimes", "Frequently", "Always"],
                    index=["no", "Sometimes", "Frequently", "Always"].index(cf_values.get('freq_alcohol', 'no')),
                    help="How often do you drink alcohol?",
                    key=f"cf_freq_alcohol{reset_suffix}",
                )

                cf_values['transport'] = st.selectbox(
                    "Usual transport mode",
                    ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
                    index=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"].index(
                        cf_values.get('transport', 'Public_Transportation')
                    ),
                    help="Your most common way of getting around.",
                    key=f"cf_transport{reset_suffix}",
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