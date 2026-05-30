

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
sonar_data = pd.read_csv("C:\Dataset\sonar_data.csv", header=None)

# Split data
X = sonar_data.drop(columns=[60])
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=1
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Streamlit UI
st.title("🚢 Sonar Rock vs Mine Predictor")

st.write(
    "Enter 60 sonar readings separated by commas."
)

user_input = st.text_input(
    "Input Data",
    ""
)

if st.button("Predict"):
    try:
        input_list = [float(x) for x in user_input.split(",")]

        if len(input_list) != 60:
            st.error("Please enter exactly 60 values.")
        else:
            input_data = np.asarray(input_list).reshape(1, -1)

            prediction = model.predict(input_data)

            if prediction[0] == 'R':
                st.success("🪨 Rock")
            else:
                st.error("💣 Mine")

    except ValueError:
        st.error("Please enter valid numeric values.")