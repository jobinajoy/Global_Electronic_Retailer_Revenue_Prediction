import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import requests
import pandas as pd
from io import StringIO, BytesIO

# Function to download and load a CSV file from a URL
def download_csv(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return pd.read_csv(StringIO(response.text))

# Function to download and load a file from a URL
def download_and_load_model(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# Load the CSV file
# URL to your CSV file on GitHub
csv_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Dataset/Global_Electronics_Retailer/Products.csv?raw=true'

# Load the CSV file
products_df = download_csv(csv_url)

# Clean numeric columns: remove '$' and convert to float
def clean_currency_column(df, column_name):
    df[column_name] = df[column_name].replace('[\$,]', '', regex=True).astype(float)

# Apply the cleaning function to relevant columns
clean_currency_column(products_df, 'Unit Cost USD')
clean_currency_column(products_df, 'Unit Price USD')

# URLs to your model files on GitHub
model_lr_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/linear_regression_model.pkl?raw=true'
elastic_net_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/elastic_net_model.pkl?raw=true'
model_lasso_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/lasso_model.pkl?raw=true'
model_ridge_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/ridge_model.pkl?raw=true'

# Download and load models
model_lr = download_and_load_model(model_lr_url)
elastic_net = download_and_load_model(elastic_net_url)
model_lasso = download_and_load_model(model_lasso_url)
model_ridge = download_and_load_model(model_ridge_url)

# Load the label encoder and scaler from GitHub
label_encoders_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/label_encoders.pkl?raw=true'
scaler_url = 'https://github.com/jobinajoy/Global_Electronic_Retailer_Revenue_Prediction/blob/dev/Saved_Models/scaler.pkl?raw=true'

label_encoders = download_and_load_model(label_encoders_url)
scaler = download_and_load_model(scaler_url)

# Define a function to make predictions
def make_prediction(model, input_data):
    return model.predict(input_data)

# Streamlit App Layout
st.title('Regression Model Prediction Interface')

st.sidebar.header('User Input Parameters')

# Define a function to make predictions and calculate metrics
def make_prediction_and_evaluate(model, input_data, y_true):
    prediction = model.predict(input_data)
    rmse = mean_squared_error(y_true, prediction, squared=False)
    r2 = r2_score(y_true, prediction)
    return prediction, rmse, r2

def user_input_features():
    # Get unique product names
    product_name = st.sidebar.selectbox('Product Name', products_df['Product Name'].unique())
    
    # Filter the dataframe based on the selected product
    filtered_df = products_df[products_df['Product Name'] == product_name]
    
    if filtered_df.empty:
        st.error("No data available for the selected product.")
        return pd.DataFrame()
    
    # Get unique categories, brands, and colors for the selected product
    category = st.sidebar.selectbox('Category', filtered_df['Category'].unique())
    brand = st.sidebar.selectbox('Brand', filtered_df['Brand'].unique())
    color = st.sidebar.selectbox('Color', filtered_df['Color'].unique())
    
    quantity = st.sidebar.slider('Quantity', 1, 100, 50)
    current_year = datetime.now().year
    order_year = st.sidebar.slider('Order Year', current_year -10 , current_year + 10, current_year)
    
    # Extract unit cost and unit price for the selected combination
    product_data = filtered_df[(filtered_df['Category'] == category) & (filtered_df['Brand'] == brand) & (filtered_df['Color'] == color)]
    
    if product_data.empty:
        st.error("No data available for the selected combination of category, brand, and color.")
        return pd.DataFrame()
    
    unit_cost = product_data['Unit Cost USD'].values[0]
    unit_price = product_data['Unit Price USD'].values[0]
    
    data = {
        'Product Name': product_name,
        'Category': category,
        'Brand': brand,
        'Color': color,
        'Quantity': quantity,
        'Order Year': order_year,
        'Unit Cost USD': unit_cost,
        'Unit Price USD': unit_price,
        # Include any additional features that were part of the model's training
        'Order Processing Time': 0,  # Placeholder, update based on how this feature should be computed
        'Total Revenue': 0  # Placeholder, this will typically be the target feature, so it might not be in the input
    }
    
    features = pd.DataFrame(data, index=[0])
    
    # Apply label encoding to categorical features
    for column in ['Product Name', 'Category', 'Brand', 'Color']:
        if column in label_encoders:
            try:
                features[column] = label_encoders[column].transform(features[column])
            except ValueError:
                # Handle unseen values by fitting a new LabelEncoder
                new_encoder = LabelEncoder()
                combined_classes = list(label_encoders[column].classes_) + list(features[column].unique())
                new_encoder.fit(combined_classes)
                features[column] = new_encoder.transform(features[column])
                label_encoders[column] = new_encoder  # Update the encoder
        else:
            # Fit a new encoder if it's missing
            new_encoder = LabelEncoder()
            features[column] = new_encoder.fit_transform(features[column])
            label_encoders[column] = new_encoder
    
    # Reorder columns to match the order used during training
    expected_features = ['Product Name', 'Category', 'Brand', 'Color', 'Quantity', 'Order Year', 
                         'Unit Cost USD', 'Unit Price USD', 'Order Processing Time', 'Total Revenue']
    
    features = features.reindex(columns=expected_features, fill_value=0)

    # Check if all necessary columns are present before scaling
    numerical_cols = ['Unit Cost USD', 'Unit Price USD', 'Order Processing Time', 'Total Revenue']
    missing_cols = [col for col in numerical_cols if col not in features.columns]
    
    if missing_cols:
        st.error(f"Missing columns in the input data: {', '.join(missing_cols)}")
        return pd.DataFrame()

    # Scale numerical features
    features[numerical_cols] = scaler.transform(features[numerical_cols])
    
    return features

input_df1 = user_input_features()

columns_to_keep = ['Product Name', 'Category','Quantity',
                   'Order Year', 'Color', 'Unit Cost USD', 'Unit Price USD','Brand']

input_df= input_df1[columns_to_keep]

if not input_df.empty:
    # Model Selection
    model_option = st.selectbox(
        'Select Model',
        ('Linear Regression', 'Elastic Net', 'Lasso Regression', 'Ridge Regression')
    )

    # Prediction
    if model_option == 'Linear Regression':
        model = model_lr
    elif model_option == 'Elastic Net':
        model = elastic_net
    elif model_option == 'Lasso Regression':
        model = model_lasso
    elif model_option == 'Ridge Regression':
        model = model_ridge
    # Placeholder for true target value (Example)
    y_true = np.array([100000000000])
    prediction, rmse, r2 = make_prediction_and_evaluate(model, input_df, y_true)

    st.write(f'### Predicted Total Revenue: ${prediction[0]:.2f}')
