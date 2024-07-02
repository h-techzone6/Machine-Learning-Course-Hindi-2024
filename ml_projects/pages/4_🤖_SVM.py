# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
data = pd.read_csv(url, names=columns)

# Separate features (X) and target variable (y)
X = data.drop('class', axis=1)
y = data['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Function to predict banknote authenticity based on user input
def predict_authenticity(variance, skewness, curtosis, entropy):
    input_data = [[variance, skewness, curtosis, entropy]]
    input_data_scaled = scaler.transform(input_data)
    prediction = svm_classifier.predict(input_data_scaled)
    return prediction[0]

# Streamlit app starts here
def main():
    # Set title and description
    st.title('Banknote Authenticity Prediction 💵💶💷')
    st.write('Use the sliders to input banknote features and predict its authenticity.')

    # Input sliders for banknote features
    variance = st.slider('Variance', min_value=-7.0, max_value=7.0, step=0.1)
    skewness = st.slider('Skewness', min_value=-13.0, max_value=13.0, step=0.1)
    curtosis = st.slider('Curtosis', min_value=-15.0, max_value=17.0, step=0.1)
    entropy = st.slider('Entropy', min_value=-8.0, max_value=2.0, step=0.1)

    # Button to predict banknote authenticity
    if st.button('Predict Authenticity'):
        # Predict banknote authenticity based on user input
        prediction = predict_authenticity(variance, skewness, curtosis, entropy)
        
        # Display predicted result
        if prediction == 0:
            st.subheader('Prediction:')
            st.write('The banknote is predicted to be Authentic.')
        else:
            st.subheader('Prediction:')
            st.write('The banknote is predicted to be Counterfeit.')

# Run the app
if __name__ == '__main__':
    main()
