# Import libraries
import pandas as pd
import numpy as np
import streamlit as st 
import joblib

# Load the model and data columns into the app
model = joblib.load("road_accident_model.pkl")
model_columns = joblib.load("road_columns.pkl")

st.set_page_config(page_title="Prediction of the likelihood of road accidents", 
                   page_icon="🧠🚗💥🚗", layout="wide")
st.title("🧠🚗💥🚗 Road Accident Likelihood Predictor")
st.markdown(
    "Predict the likelihood of occurrence of a road accident."
    "Get insights from the predictions and follow guidelines to reduce the risk of road accidents."
)

# Sidebar inputs
st.sidebar.header("Kindly provide the following details:")

road_type = st.sidebar.selectbox("Select the Road Type", ['Highway', 'Rural', 'Urban'])
num_lanes = st.sidebar.number_input("How many lanes are on the road?", min_value=0, step=1)
curvature = st.sidebar.number_input("Input the curvature value of the road [0-1]", min_value=0.0, step=0.01, max_value=1.0)
speed_limit = st.sidebar.number_input("Input the speed limit on the road in km/h", min_value=0, step=1)
lighting = st.sidebar.selectbox("What is the lighting condition on the road?", ['Daylight', 'Dim', 'Night'])
weather = st.sidebar.selectbox("What is the weather condition on the road?", ['Clear', 'Rainy', 'Foggy'])
road_sign_present = st.sidebar.selectbox("Road signs are present", [True, False])
time_of_day = st.sidebar.selectbox("What is the time of the day?", ['Morning', 'Afternoon', 'Evening'])
public_road = st.sidebar.selectbox("It is a public road", [True, False])
holiday = st.sidebar.selectbox("It is a holiday", [True, False])
school_season = st.sidebar.selectbox("It is a school season", [True, False])
num_reported_accidents = st.sidebar.number_input('What is the number of reported accidents on this road?', min_value=0, step=1)

# Create dataframe
input_df = pd.DataFrame({
    'road_type': [road_type],
    'num_lanes': [num_lanes],
    'curvature': [curvature],
    'speed_limit': [speed_limit],
    'lighting': [lighting],
    'weather': [weather],
    'road_sign_present': [road_sign_present],
    'public_road': [public_road],
    'time_of_day': [time_of_day],
    'holiday': [holiday],
    'school_season': [school_season],
    'num_reported_accidents': [num_reported_accidents]
})

# Base risk calculation
input_df["base_risk"] = (
    0.3 * input_df["curvature"] +
    0.2 * (input_df["lighting"].str.lower() == "night").astype(int) +
    0.1 * (input_df["weather"].str.lower() != "clear").astype(int) +
    0.2 * (input_df["speed_limit"] >= 60).astype(int) +
    0.1 * (input_df["num_reported_accidents"] > 2).astype(int)
)

# Feature encoding
categorical_cols = ['weather','road_type','time_of_day']

# weather
input_df['weather_clear'] = input_df['weather'] == 'clear'
input_df['weather_foggy'] = input_df['weather'] == 'foggy'
input_df['weather_rainy'] = input_df['weather'] == 'rainy'

# Road type
input_df['road_type_highway'] = input_df['road_type'] == 'highway'
input_df['road_type_rural'] = input_df['road_type'] == 'rural'
input_df['road_type_urban'] = input_df['road_type'] == 'urban'

# time of the day
input_df['time_of_day_afternoon'] = input_df['time_of_day'] == 'afternoon'
input_df['time_of_day_evening'] = input_df['time_of_day'] == 'evening'
input_df['time_of_day_morning'] = input_df['time_of_day'] == 'morning'


# Lighting encoding
lighting_mapping = {'daylight': 0, 'dim': 0, 'night': 1}
input_df['lighting_encoded'] = input_df['lighting'].str.lower().map(lighting_mapping)

# Convert boolean columns to int
binary_cols = ['holiday', 'school_season', 'road_sign_present', 'public_road']
for col in binary_cols:
    input_df[col] = input_df[col].astype(int)

# Feature engineering
input_df['speed_curvature'] = input_df['speed_limit'] * input_df['curvature']
input_df['lanes_curvature'] = input_df['num_lanes'] * input_df['curvature']
input_df['speed_lighting'] = input_df['speed_limit'] * input_df['lighting_encoded']
input_df['curvature_lighting'] = input_df['curvature'] * input_df['lighting_encoded']
input_df['high_speed_curve'] = ((input_df['speed_limit'] >= 60) & (input_df['curvature'] > 0.5)).astype(int)
input_df['curvature_squared'] = input_df['curvature'] ** 2
input_df['speed_squared'] = input_df['speed_limit'] ** 2
input_df['bad_conditions'] = (((input_df.get('weather_Rainy', 0) == 1) | (input_df.get('weather_Foggy', 0) == 1)) &
                            (input_df['lighting_encoded'] == 1)).astype(int)

# Drop unnecessary columns
final_df = input_df.drop(columns=['lighting'], errors='ignore')

# Align input columns with training columns
final_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
st.subheader("Road Accident Likelihood Prediction")
if st.button("🧠🚗💥🚗 Predict Road Accident Likelihood"):
    prediction = model.predict(final_df)
    st.success(f"⚠️ Predicted Accident Likelihood: **{prediction[0]:.3f}**")

    if prediction < 0.4:
        st.info("Likelihood is low, but always follow road safety measures.")
    elif prediction < 0.7:
        st.warning("Likelihood is moderate, please be careful on the road.")
    else:
        st.error("Likelihood is high! Strictly adhere to all road safety measures.")
    
    st.balloons()

# Additional information
st.markdown("---")
st.subheader("💡 About This App")
st.write("""
This app predicts the likelihood of a road accident occurring on a given road.Road accidents are a major
cause of injuries and loss of lives worldwide. 
They often stem from factors such as speeding, distracted driving, poor road conditions, or failure to
follow traffic regulations.
Raising awareness, improving road infrastructure, and practicing safe driving can significantly reduce 
these incidents and save lives. 
One critical step toward prevention is the ability to predict the likelihood of accidents in advance, allowing
motorists and authorities to take proactive measures.
This app is designed to support road safety efforts by increasing awareness and helping users identify
potential risks, ultimately reducing the chances of accidents.
""")
