# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the Wine Quality dataset (example dataset)
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    wine_data = pd.read_csv('winequality-red.csv')
    return wine_data

wine_data = load_data()

# Define predictors (features) and target variable
X = wine_data.drop('quality', axis=1)  # Features: all columns except 'quality'
y = wine_data['quality']  # Target variable: 'quality'

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X, y)  # Fit the model on the entire dataset

# Function to predict wine quality based on user input
def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, pH, sulphates, alcohol):
    
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                   chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                   density, pH, sulphates, alcohol]]
    
    predicted_quality = model.predict(input_data)
    return predicted_quality[0]

# Streamlit app starts here
def main():
    # Set title and description
    st.title('Wine Quality Prediction')
    st.write('Enter wine properties to predict the quality.🍷')

    # Input sliders for wine properties
    fixed_acidity = st.slider('Fixed Acidity', min_value=4.0, max_value=16.0, step=0.1)
    volatile_acidity = st.slider('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01)
    citric_acid = st.slider('Citric Acid', min_value=0.0, max_value=1.0, step=0.01)
    residual_sugar = st.slider('Residual Sugar', min_value=0.0, max_value=20.0, step=0.1)
    chlorides = st.slider('Chlorides', min_value=0.0, max_value=1.0, step=0.01)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', min_value=1, max_value=72, step=1)
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', min_value=6, max_value=289, step=1)
    density = st.slider('Density', min_value=0.87, max_value=1.0, step=0.001)
    pH = st.slider('pH', min_value=2.72, max_value=4.01, step=0.01)
    sulphates = st.slider('Sulphates', min_value=0.33, max_value=2.0, step=0.01)
    alcohol = st.slider('Alcohol', min_value=8.4, max_value=14.9, step=0.1)

    # Button to predict wine quality
    if st.button('Predict Wine Quality'):
        # Predict wine quality based on user input
        predicted_quality = predict_quality(fixed_acidity, volatile_acidity, citric_acid,
                                            residual_sugar, chlorides, free_sulfur_dioxide,
                                            total_sulfur_dioxide, density, pH, sulphates, alcohol)
        
        # Display predicted wine quality
        st.subheader('Predicted Wine Quality:')
        st.write(f'The predicted quality of the wine is {predicted_quality}.')

# Run the app
if __name__ == '__main__':
    main()
