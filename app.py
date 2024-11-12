import streamlit as st
import pickle
import numpy as np

# Load the trained model and label encoder from pickle
with open('rank_prediction_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Title of the app
st.title('PUBG Rank Prediction')

# Description of the app
st.write("""
This application predicts a player's rank based on their performance in PUBG.
Enter the following stats, and the model will predict the player's rank.
""")

# User input fields for relevant features
kills = st.number_input("Kills", min_value=0, value=100)
wins = st.number_input("Wins", min_value=0, value=10)
damage_dealt = st.number_input("Damage Dealt", min_value=0, value=5000)
time_survived = st.number_input("Time Survived (minutes)", min_value=0, value=40) * 60

# Create a button to make predictions
if st.button('Predict Rank'):
    # Prepare the input data in the correct shape for the model
    input_data = np.array([[kills, wins, damage_dealt, time_survived]])

    # Make the prediction using the model pipeline
    predicted_rank_encoded = model_pipeline.predict(input_data)

    # Decode the numeric prediction back to the rank label
    predicted_rank = label_encoder.inverse_transform(predicted_rank_encoded)

    # Display the predicted rank
    st.write(f"Predicted Rank: **{predicted_rank[0]}**")

