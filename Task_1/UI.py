import pickle
import streamlit as st
import pandas as pd

# Load the pre-trained classifier and scaler from the pickle file
with open('gaussian_process_models.pickle', 'rb') as f:
    data = pickle.load(f)
    classifier = data['classifier_model']
    scaler = data['scaler']

print("Model loaded successfully:")
print(classifier)

def main():
    st.set_page_config(page_title="Obesity Level Prediction Using Gaussian Process", layout="centered")

    # Custom CSS for global styling and hover effect
    st.markdown(f"""
        <style>
        /* Overall background color and font settings */
        body {{
            background-color: #24293E;
            color: #8EBBFF;
        }}
        /* Title and subtitle styling */
        .stApp .stMarkdown h1, .stApp .stMarkdown p {{
            color: #8EBBFF;
            text-align: center;
            padding: 0;
            margin: 5px 0;
        }}
        /* Input fields styling */
        .stApp .stNumberInput input {{
            background-color: #F4F5FC;
            color: #24293E;
            border-radius: 10px;
            padding: 5px;
            border: 1px solid #CCCCCC;
            transition: border-color 0.3s, box-shadow 0.3s;
        }}
        /* Bolder border when focused */
        .stApp .stNumberInput input:focus {{
            outline: none;
            border-color: #8EBBFF;
            border-width: 8px;
            box-shadow: 0 0 10px #8EBBFF;
        }}
        /* Button styling */
        .stButton > button {{
            background-color: #8EBBFF;
            color: #24293E;
            border-radius: 10px;
            padding: 8px 15px;
            border: none;
            transition: background-color 0.3s;
            margin-top: 10px;
        }}
        .stButton > button:hover {{
            background-color: #6BA8E5;
        }}
        /* Container padding and margin adjustments */
        .stApp .stMarkdown, .stApp .stButton {{
            margin-top: 5px;
            padding: 5px;
        }}
        /* Reduce the width and height of the entire container */
        .block-container {{
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            max-width: 750px;
            margin: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

    st.title('Obesity Level Prediction Using Gaussian Process')
    st.markdown("""
        This application predicts the obesity level of an individual based on various health and lifestyle factors.
        Please enter the relevant information to get the prediction.
    """)

    # First row: Height and Age
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input('Height (cm)', min_value=0.0, max_value=250.0, step=0.1, format="%.1f", value=None)
    
    with col2:
        age = st.number_input('Age (years)', min_value=0, max_value=100, step=1, value=None)
    
    # Second row: Weight and Frequency of vegetables consumption
    col3, col4 = st.columns(2)
    
    with col3:
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=200.0, step=0.1, format="%.1f", value=None)

    with col4:
        frequency = st.number_input('Frequency of consumption of vegetables (1-3)', min_value=1, max_value=3, step=1, value=1)

    if st.button('Predict'):
        if height is None or weight is None or age is None:
            st.error("Please enter valid values for height, weight, and age.")
        else:
            # Convert Height from cm to meters
            height_meters = height / 100.0  # Convert cm to meters

            # Create the input DataFrame
            input_features = pd.DataFrame([[age, height_meters, weight, frequency]], 
                                        columns=['Age', 'Height', 'Weight', 'FCVC'])
            
            # Debug: Print input features before scaling
            print("Feature Names in UI Input:")
            print(input_features.columns)
            print("Input Features Before Scaling:")
            print(input_features)

            # Scale the input features using the scaler used in training
            input_features_scaled = scaler.transform(input_features)
            
            # Debug: Print scaled features
            print("Scaled Features After Scaling:")
            print(input_features_scaled)

            # Make a prediction using the classifier
            prediction = classifier.predict(input_features_scaled)

            obesity_levels = {
                0: 'Insufficient Weight', 
                1: 'Normal Weight', 
                2: 'Overweight Level I', 
                3: 'Overweight Level II', 
                4: 'Obesity Type I', 
                5: 'Obesity Type II', 
                6: 'Obesity Type III'
            }
            st.success(f'The predicted obesity level is {obesity_levels[prediction[0]]}.')

    st.markdown("""
        ---
        Developed by Ayush Rayamajhi & Kaushal Rijal
    """)

if __name__ == '__main__':
    main()


