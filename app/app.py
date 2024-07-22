import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

# Function to load the model and data
def load_model_and_data(notebook_path, data_path):
    # Load the model (assuming it's saved as a joblib file)
    model = load(os.path.join(notebook_path, 'tennis_model.joblib'))
    
    # Load the data
    data = pd.read_csv(os.path.join(data_path, 'tennis_data.csv'))
    
    return model, data

# Function to get unique values for dropdown lists
def get_unique_values(data):
    return {
        'Player 1 Name': sorted(data['player_1_name'].unique()),
        'Player 2 Name': sorted(data['player_2_name'].unique()),
        'Surface': sorted(data['surface'].unique()),
        'Tourney Level': sorted(data['tourney_level'].unique())
    }

# Function to get latest player data
def get_latest_player_data(data, player_name):
    player_data = data[
        (data['player_1_name'] == player_name) | 
        (data['player_2_name'] == player_name)
    ].sort_values('date').iloc[-1]
    
    if player_data['player_1_name'] == player_name:
        return {
            'rank': player_data['player_1_rank'],
            'elo': player_data['player_1_elo'],
            'matches': player_data['player_1_matches'],
            'age': player_data['player_1_age']
        }
    else:
        return {
            'rank': player_data['player_2_rank'],
            'elo': player_data['player_2_elo'],
            'matches': player_data['player_2_matches'],
            'age': player_data['player_2_age']
        }

# Function to predict the match outcome
def predict_match(model, input_data):
    prediction = model.predict_proba(input_data)[0]
    player_2_wins = prediction[1] > 0.5
    return player_2_wins, prediction[1] * 100

# Main Streamlit app
def main():
    st.title("Tennis Match Prediction")

    # File uploader for notebook and data directory
    notebook_path = st.sidebar.file_uploader("Select Jupyter Notebook", type="ipynb")
    data_path = st.sidebar.text_input("Enter path to data directory")

    if notebook_path and data_path:
        model, data = load_model_and_data(notebook_path, data_path)
        unique_values = get_unique_values(data)

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Match Information")
            player1 = st.selectbox("Player 1 Name", unique_values['Player 1 Name'])
            player2 = st.selectbox("Player 2 Name", unique_values['Player 2 Name'])
            surface = st.selectbox("Surface", unique_values['Surface'])
            tourney_level = st.selectbox("Tourney Level", unique_values['Tourney Level'])

        if st.button("Predict"):
            # Get latest player data
            player1_data = get_latest_player_data(data, player1)
            player2_data = get_latest_player_data(data, player2)

            # Display player stats
            with col1:
                st.subheader("Player Statistics")
                st.write(f"**{player1}**")
                st.write(f"Ranking: {player1_data['rank']}")
                st.write(f"Elo Rating: {player1_data['elo']}")
                st.write(f"Matches Played: {player1_data['matches']}")
                st.write(f"Age: {player1_data['age']}")

                st.write(f"**{player2}**")
                st.write(f"Ranking: {player2_data['rank']}")
                st.write(f"Elo Rating: {player2_data['elo']}")
                st.write(f"Matches Played: {player2_data['matches']}")
                st.write(f"Age: {player2_data['age']}")

            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'player_1_rank': [player1_data['rank']],
                'player_2_rank': [player2_data['rank']],
                'player_1_elo': [player1_data['elo']],
                'player_2_elo': [player2_data['elo']],
                'player_1_matches': [player1_data['matches']],
                'player_2_matches': [player2_data['matches']],
                'player_1_age': [player1_data['age']],
                'player_2_age': [player2_data['age']],
                'surface': [surface],
                'tourney_level': [tourney_level]
            })

            # Make prediction
            player_2_wins, win_probability = predict_match(model, input_data)

            # Display prediction results
            with col2:
                st.subheader("Prediction Results")
                winner = player2 if player_2_wins else player1
                st.write(f"**Predicted Winner: {winner}**")

                # Create bar chart
                fig, ax = plt.subplots()
                players = [player1, player2]
                probabilities = [100 - win_probability, win_probability]
                ax.bar(players, probabilities)
                ax.set_ylabel("Win Probability (%)")
                ax.set_title("Match Outcome Probability")

                # Display the chart
                st.pyplot(fig)

if __name__ == "__main__":
    main()