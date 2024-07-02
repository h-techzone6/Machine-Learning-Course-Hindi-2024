# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the dataset
data = {
    'Age': [25, 30, 35, 20, 40, 45, 18, 50, 22, 36],
    'Income': [50000, 70000, 90000, 30000, 100000, 110000, 20000, 120000, 35000, 80000],
    'Purchased': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 0 = No, 1 = Yes
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the entire dataset
model.fit(df[['Age', 'Income']], df['Purchased'])

# Define a function to predict purchase based on input values
def predict_purchase(age, income):
    input_data = np.array([[age, income]])
    predicted_purchase = model.predict(input_data)
    return predicted_purchase[0]

# Streamlit app starts here
def main():
    # Set title and description
    st.title('Customer Purchase Prediction 💲')
    st.write('Enter customer\'s age and income to predict whether they will make a purchase.')

    # Input fields for age and income
    age = st.text_input('Age')
    income = st.text_input('Income')

    # Button to predict purchase
    if st.button('Predict Purchase'):
        # Validate input
        if (age and income):
            # Convert input to integers
            try:
                age = int(age)
                income = int(income)
            except ValueError:
                st.error('Please enter valid numeric values.')
                return

            # Predict purchase based on user input
            predicted_purchase = predict_purchase(age, income)

            # Display predicted purchase
            if predicted_purchase == 1:
                st.subheader('Predicted Purchase:')
                st.write('Customer is likely to make a purchase.')
            else:
                st.subheader('Predicted Purchase:')
                st.write('Customer is unlikely to make a purchase.')
        else:
            st.warning('Please enter values for both age and income.')

# Run the app
if __name__ == '__main__':
    main()
