import streamlit as st
import pickle

st.markdown("# Predictive Analytics 🎯")

st.sidebar.markdown("# Predictive Analytics 🎯")

"""
⚠️ Add here some predictive analytics with Widgets and Plots
"""

st.write("# Example of model prediction")

# Load model
pre_trained_model_path = "./assets/trained_model.pickle"
loaded_model = None # This will be replaced by the trained model in the pickle 

with open(pre_trained_model_path, "rb") as readFile:
    loaded_model = pickle.load(readFile)


# COLUMNS
left_column, right_column = st.columns(2)

user_data = []
# Call Streamlit functions inside a "with" block to keep it in a column:
with left_column:
    length = st.slider("Sepal Length", min_value=4.0, max_value=9.0, value = 5.0)
with right_column:
    width = st.slider("Sepal Width", min_value=1.5, max_value=4.0, value = 3.0)

if st.button('Predict!'):
    user_data = [[length, width]]
    prediction = loaded_model.predict(user_data)
    st.write(f"The predicted value for data {user_data} is {prediction}")

"""
# 
⚠️ Add some visualizations to help understanding what the predictions mean...
"""