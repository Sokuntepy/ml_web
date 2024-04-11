import pandas as pd
import os
import pickle
import numpy as np
import streamlit as st
import csv
import base64  

#loading the saved model
loaded_model = pickle.load(open('train_model.sav','rb'))

# Function to save user inputs and predictions
def save_prediction_history(inputs, prediction):
    history_file = 'prediction_history.csv'
    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Prediction'])
    with open(history_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(inputs + [prediction])

# Add the function to generate download link for CSV file
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_histories.csv">Download CSV File</a>'
    return href

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

# Add a predict button
if st.button('Predict'):
    # Create a feature vector from the input data
    feature_vector = [pclass, sex_encoded, age, sibsp, parch, float(fare), embarked_options.index(embarked)]

    # Convert the feature vector to a NumPy array
    feature_vector = np.array(feature_vector).reshape(1, -1)  # Reshape to match the expected input shape of the model

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(feature_vector)

    # Save the prediction history
    save_prediction_history([pclass, sex, age, sibsp, parch, float(fare), embarked, 'Survived' if prediction[0] == 1 else 'Not Survived'], prediction[0])

    # Display the prediction result
    if prediction[0] == 1:
        st.write('The passenger is predicted to have **survived** the Titanic incident.')
        st.image('/Users/macbookpro/Desktop/Titanic_Survivors_Prediction/survive.png')
    else:
        st.write('The passenger is predicted to have **not survived** the Titanic incident.')
        st.image('/Users/macbookpro/Desktop/Titanic_Survivors_Prediction/drown.jpg')

# Add download button
if st.button('Download Prediction History'):
    history_df = pd.read_csv('prediction_history.csv')
    # Replace encoded values with labels
    history_df['Sex'] = history_df['Sex'].apply(lambda x: 'female' if x == 1 else 'male')
    embarked_label = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    history_df['Embarked'] = history_df['Embarked'].map(embarked_label)
    history_df['Prediction'] = history_df['Prediction'].apply(lambda x: 'Survived' if x == 1 else 'Not Survived')
    st.write(history_df)
    st.markdown(get_table_download_link(history_df), unsafe_allow_html=True)
