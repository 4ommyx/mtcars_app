import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('mtcar_linear_reg.pkl')

st.set_page_config(
    page_title="Car Prediction App",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Car Prediction App")
st.markdown("""
Welcome to the car prediction app !    
Enter the required features below, and click **Predict** to see the result.
""")

st.markdown("""
### Model Performance
Mean Squared Error : 6.251           
R² Score : 0.835
""")

st.markdown("""
### Description of Feature in :red-background[mtcars] Dataset

- **:red-background[mpg (Miles per Gallon)]**  
  Indicates the fuel efficiency of the car, measured in miles per gallon. Higher values represent better fuel economy.

- **:red-background[wt (Weight)]**  
  The weight of the car in 1000 lbs. Heavier cars generally consume more fuel.

- **:red-background[cyl (Number of Cylinders)]**  
  The number of cylinders in the car's engine. Common values are 4, 6, and 8. More cylinders typically provide more power but lower fuel efficiency.

- **:red-background[disp (Displacement)]**  
  The total volume of all the cylinders in the engine, measured in cubic inches. A higher displacement indicates a larger engine size.

- **:red-background[hp (Horsepower)]**  
  The power output of the engine, measured in horsepower. A higher value indicates a more powerful engine.

- **:red-background[drat (Driveshaft Ratio)]**  
  The ratio of the rear axle to the driveshaft. This affects the car's acceleration and fuel economy.

- **:red-background[vs (Engine Shape)]**  
  Describes the engine shape:
  - :red-background[:red[0]] : V-shaped engine
  - :red-background[:red[1]] : Straight engine

""")

# Input Features
st.write("### Enter Features:")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

feature_wt = col1.number_input(
    label="Feature 1 (Weight):", 
    min_value=1.0, 
    max_value=6.0, 
    value=1.0, 
    step=0.1, 
    format="%.2f", 
    help="Enter the weight in 1000 lbs.",
    label_visibility="visible"
)
feature_cyl = col5.slider(
    label="Feature 5 (Cylinders):", 
    min_value=4, 
    max_value=8, 
    value=6, 
    step=2, 
    help="Choose the number of cylinders."
)
feature_disp = col2.number_input(
    label="Feature 2 (Displacement):", 
    min_value=50, 
    max_value=500, 
    value=50, 
    step=1, 
    format="%d", 
    help="Enter the engine displacement (cubic inches)."
)
feature_hp = col3.number_input(
    label="Feature 3 (Horsepower):", 
    min_value=50, 
    max_value=450, 
    value=50, 
    step=1, 
    format="%d", 
    help="Enter the engine horsepower."
)
feature_drat = col4.number_input(
    label="Feature 4 (Driveshaft Ratio):", 
    min_value=2.0, 
    max_value=5.0, 
    value=2.0, 
    step=0.1, 
    format="%.2f", 
    help="Enter the driveshaft ratio."
)
options = ["V-shaped engine : 0", "Straight engine : 1"]
selection = col6.selectbox(
    label="Feature 6 (Engine Shape):",
    options=options,
    help="Choose the engine shape."
)

if st.button("🔮 Predict"):
    # st.markdown([feature_wt, feature_cyl, feature_disp, feature_hp, feature_drat, options.index(selection)])
    try:
        features = np.array([[feature_wt, feature_cyl, feature_disp, feature_hp, feature_drat, options.index(selection)]])
        prediction = model.predict(features)
        st.success(f"🎉 The predicted value is: **{prediction[0][0].round(3)}**")
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")


# Add a footer with credits
st.markdown("""
---
💡 *Built with Streamlit* | 🚀 *Your predictive companion*
""")
