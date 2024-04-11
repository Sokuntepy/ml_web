import pandas as pd
import os
import pickle
import numpy as np
import streamlit as st
import csv
import base64  

#loading the saved model
loaded_model = pickle.load(open('train_model.sav','rb'))

# Display image URLs
survive_image_url = "https://raw.githubusercontent.com/Sokuntepy/ml_web/main/survive.png"
titanic_image_url = "https://raw.githubusercontent.com/Sokuntepy/ml_web/main/titanic.jpg"
drown_image_url = "https://raw.githubusercontent.com/Sokuntepy/ml_web/main/drown.jpg"

# Create a two-column layout
col1, col2 = st.columns([1, 3])

# Display the image in the first column
with col1:
    st.image(titanic_image_url, width=130)

# Display the title and description in the second column
with col2:
    st.markdown("<h1 style='color: blue; font-size: 30px;'>Titanic Survivors Prediction</h1>", unsafe_allow_html=True)
    st.write('Please enter each passenger information to predict whether they survived in Titanic incident or not')

# Define input widgets for each variable
pclass_options = [1, 2, 3]
pclass = st.selectbox('Passenger Class (Pclass)', pclass_options)

sex = st.radio('Sex', ['male', 'female'])
sex_encoded = 1 if sex == 'female' else 0

age = st.number_input('Age', value=30)

sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', value=1)

parch = st.number_input('Number of Parents/Children Aboard (Parch)', value=2)

fare = st.radio('Fare', ['7', '12', '30', '105'])
st.write('💷 Fare is measured in pound in 1972 ranking from each class, which are third class, second class, first-class berths, and first-class suite')

embarked_options = ['C', 'Q', 'S']
embarked = st.selectbox('Embarked', embarked_options)
st.write(':bulb: C: Cherbourg, S: Southampton, Q: Queenstown')

# Store history of predictions
history = []

# Add a predict button
if st.button('Predict'):
    # Create a feature vector from the input data
    feature_vector = [pclass, sex_encoded, age, sibsp, parch, float(fare), embarked_options.index(embarked)]

    # Convert the feature vector to a NumPy array
    feature_vector = np.array(feature_vector).reshape(1, -1)  # Reshape to match the expected input shape of the model

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(feature_vector)

    # Store input and prediction in history
    history.append({
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Prediction': 'Survived' if prediction[0] == 1 else 'Not Survived'
    })

    # Display the prediction result
    if prediction[0] == 1:
        st.write('The passenger is predicted to have **survived** the Titanic incident.')
        st.markdown(f'<img src="{survive_image_url}" alt="Survived" style="width: 200px;">', unsafe_allow_html=True)
    else:
        st.write('The passenger is predicted to have **not survived** the Titanic incident.')
        st.markdown(f'<img src="{drown_image_url}" alt="Not Survived" style="width: 200px;">', unsafe_allow_html=True)

# Allow users to download history in Excel format
if len(history) > 0:
    history_df = pd.DataFrame(history)
    st.write('### Prediction History:')
    st.write(history_df)

    csv = history_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download Prediction History</a>'
    st.markdown(href, unsafe_allow_html=True)
