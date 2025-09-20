import streamlit as st
import pickle
import pandas as pd

# List of teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl','rb'))

# App title
st.title(' WinMantra ')
st.header('  IPL Win Predictor ')

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# City selection
selected_city = st.selectbox('Select host city', sorted(cities))

# Target score
target = st.number_input('Target', min_value=0, step=1)

# Score, overs, wickets
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Predict button
if st.button('Predict Probability'):
    if overs == 0:
        st.error("Overs cannot be zero!")
    else:
        runs_left = target - score
        balls_left = max(0, 120 - int(overs*6))  # T20 match
        wickets_left = max(0, 10 - wickets_out)
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

        # Prepare input for pipeline
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.header(f"{batting_team} - {round(win*100)}%")
        st.header(f"{bowling_team} - {round(loss*100)}%")
