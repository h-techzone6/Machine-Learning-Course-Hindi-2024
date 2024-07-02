# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'YearsExperience': [1, 3, 5, 7, 9, 11, 13, 17, 19, 21],
    'EducationLevel': [12, 13, 14, 15, 16, 18, 20, 20, 22, 24],
    'Salary': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Split data into predictors (X) and target (y)
X = df[['YearsExperience', 'EducationLevel']]  # predictors
y = df['Salary']  # target

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Define a function to predict salary based on input values
def predict_salary(years_exp, education):
    new_data = pd.DataFrame({
        'YearsExperience': [years_exp],
        'EducationLevel': [education]
    })
    predicted_salary = model.predict(new_data)
    return predicted_salary[0]

# Streamlit app starts here
def main():
    # Set title and description
    st.title('Salary Prediction App 💼')
    st.write('Enter the number of years of experience and education level to predict the salary.')

    # Input fields for years of experience and education level
    years_exp = st.text_input('Years of Experience')
    education = st.text_input('Education Level (years)')

    # Button to predict salary
    if st.button('Predict Salary'):
        # Validate input
        if years_exp and education:
            # Convert input to integers
            try:
                years_exp = int(years_exp)
                education = int(education)
            except ValueError:
                st.error('Please enter valid numeric values.')
                return

            # Predict salary based on user input
            predicted_salary = predict_salary(years_exp, education)

            # Display predicted salary
            st.subheader('Predicted Salary:')
            st.write(f'Rs {predicted_salary:.2f}')
        else:
            st.warning('Please enter values for both fields.')

# Run the app
if __name__ == '__main__':
    main()
