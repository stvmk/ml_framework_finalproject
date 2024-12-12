import pandas as pd
from cuml.model_selection import train_test_split
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
from cuml.svm import SVR as cuSVR
from cuml.neighbors import KNeighborsRegressor as cuKNNRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import logging
import os
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load the dataset
def load_dataset(file_path):
    try:
        logging.info("Loading dataset...")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise e

# Function to preprocess the dataset
def preprocess_data(df):
    logging.info("Preprocessing dataset...")
    # Encode 'IsHoliday' column
    encoder = LabelEncoder()
    df['IsHoliday'] = encoder.fit_transform(df['IsHoliday'])

    # Convert 'Date' column to datetime format and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.drop(columns=['Date'], inplace=True)

    # Convert all columns to numeric and handle non-numeric values
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df[col] = LabelEncoder().fit_transform(df[col])
        if df[col].dtype != 'object':
            df[col] = df[col].astype('float32')

    return df

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": cuLinearRegression(),
        "Random Forest": cuRandomForestRegressor(n_estimators=100, random_state=42),
        "K-Nearest Neighbors (KNN)": cuKNNRegressor(n_neighbors=5),
    }

    results = {}
    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[model_name] = mse

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[:200], color='blue', linestyle='--', label='Actual Values')
        plt.plot(predictions[:200], color='red', linestyle='--', label='Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Weekly Sales')
        plt.title(f'{model_name}: Actual vs Predicted Weekly Sales')
        plt.legend()
        plt.show()

    return results

dataset_path = os.getenv('DATASET_PATH', 'train.csv')  # Use environment variable or default path
train_dataset = load_dataset(dataset_path)
train_dataset = preprocess_data(train_dataset)

    # Define features and target variable
X = train_dataset.drop(columns=['Weekly_Sales']).values
y = train_dataset['Weekly_Sales'].values

    # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Display results
for model_name, mse in results.items():
    logging.info(f"{model_name}: Mean Squared Error = {mse}")

plt.figure(figsize=(10.6, 6))
plt.bar(results.keys(), results.values())
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance')
plt.show()

