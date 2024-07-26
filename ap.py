import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the modelrrr
model = pk.load(open('model.pkl', 'rb'))

# Apply CSS for the entire page
st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    body {
        background-color: white;
        color: white;
        font-family: Arial, sans-serif;
    }
    
    /* Streamlit widgets styling */
    .stSelectbox > div, .stSlider > div, .stButton > button {
        background-color:blue;
        color: white;
        border: 1px solid black;
    }
    
    /* Styling for the button */
    .stButton > button {
        background-color: black; /* Button color */
        border: none;
        color: blue;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    
    /* Custom styling for slider */
    .stSlider > div {
        background-color: #f0f0f0;
        color: black;
    }
    
    /* Custom styling for select boxes */
    .stSelectbox > select {
        background-color: #333;
        color: black;
    }
    
    .custom-header {
        background-color: black; /* Black background for the header */
        color:pink; /* Think pink color for the title text */
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 6px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st

# Custom CSS for header
st.markdown("""
    <style>
    .header {
        background-color: white; /* Black background */
        color: red; /* Pink text */
        padding: 10px;
        text-align: center;
        font-size: 2em;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)





st.image(r'C:\Users\laxmi\Downloads\car_price\images\R.jpg', use_column_width=1000)


st.markdown('<div class="custom-header"> Car Price Prediction ML Model</div>', unsafe_allow_html=True)

# Load the car data
cars_data = pd.read_csv(r'C:\Users\laxmi\Downloads\car_price\Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit widgets
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

if st.button("Predict"):
    # Prepare the input data
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical variables
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'],
                                      [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                     list(range(1, 32)), inplace=True)

    # Predict the car price
    car_price = model.predict(input_data_model)

    # Display the result
    st.markdown('Car Price is going to be ' + str(car_price[0]))
