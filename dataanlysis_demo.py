import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import warnings 

warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load datasets
file_path = r'C:\Users\91742\Documents\final data set\diabetes.csv'
df = pd.read_csv(file_path)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Assistant',
                           ['Select an option', 'Predict', 'Dataset Analysis', 'Diabetes vs Age'],
                           icons=['', 'heart', 'activity', 'person'],
                           default_index=0)
    
   
if selected == 'Select an option':
    st.markdown('<h1 style="font-size: 36px;">**Welcome to Diabetes Prediction System**</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 24px;">This application is designed to help predict diabetes based on user input and analyze a diabetes dataset.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 24px;">Please select an option from the sidebar to get started.</p>', unsafe_allow_html=True)
    
    
if selected == 'Predict':

    # Page title
    st.title('Enter Details To Predict')

    # Getting the input data from the user
    col1, col2 = st.columns(2) 

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col1:
        BloodPressure = st.text_input('Blood Pressure value')

    with col2:
        SkinThickness = st.text_input('Skin Thickness value')

    with col1:
        Insulin = st.text_input('Insulin Level')

    with col2:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        
            # Convert inputs to float
            user_input = [float(x) for x in user_input]
            st.write(df.describe())

            # Define features (X) and target (Y)
            X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
            Y = df['Outcome']

            # Split data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,random_state=4 )

            # Train the Random Forest model    
            model = RandomForestClassifier(n_estimators=100, random_state=10)
            model.fit(X_train, Y_train)
        
            # Predict on the test data
            Y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(Y_test, Y_pred)

            # Predict on the user input 
            diab_prediction = model.predict([user_input])
        
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
                
                # Generate feature importance plot
                importances = model.feature_importances_
                feature_names = ['Number of Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 'Insulin Level', 'BMI', 'Diabetes Pedigree Function', 'Age']
                fig, ax = plt.subplots(figsize=(10, 6))  
                ax.bar(range(len(importances)), importances)  
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(feature_names, rotation=90) 
                plt.tight_layout()  
                ax.set_xlabel('Features')
                ax.set_ylabel('Feature Importance')
                st.pyplot(fig)
                
            else:
                diab_diagnosis = 'The person is not diabetic'
                
            st.success(diab_diagnosis)
            st.write("Accuracy:", accuracy*100)
        
        except ValueError as e:
            st.error(f"Please enter valid numeric values. Error: {e}")

if selected == 'Dataset Analysis':
 
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(7, 7))  
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, ax=ax)
    #ax.set_title('')

    # Display the heatmap in the Streamlit UI
    st.pyplot(fig)
    
if selected == 'Diabetes vs Age':
   
    age_groups = df['Age'].unique()
    outcome_counts_positive = []
    outcome_counts_negative = []
    for age in age_groups:
        age_group_data = df[df['Age'] == age]
        outcome_count_positive = age_group_data[age_group_data['Outcome'] == 1].shape[0]
        outcome_count_negative = age_group_data[age_group_data['Outcome'] == 0].shape[0]
        outcome_counts_positive.append(outcome_count_positive)
        outcome_counts_negative.append(outcome_count_negative)

    # Create a pyramid plot
    fig, ax = plt.subplots(figsize=(7, 4))  # Changed size to (16, 14)
    ax.barh(age_groups, outcome_counts_positive, label='Positive Outcome')
    ax.barh(age_groups, -np.array(outcome_counts_negative), label='Negative Outcome')
    ax.set_xlabel('Outcome Count')
    ax.set_ylabel('Age Groups')
    ax.set_title('Outcome vs Age')
    ax.legend()

    # Display the plot in the Streamlit UI
    st.pyplot(fig)  

