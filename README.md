# Walmart Store Sales Forecasting Web App

## Overview
This project is a web application built using Flask that forecasts store sales based on historical data from the Walmart Recruiting - Store Sales Forecasting competition on Kaggle. We implemented machine learning models such as Linear Regression, Random Forest, and K-Nearest Neighbors (KNN) to predict future sales.

## Features
- **Sales Forecasting**: Predict future sales for Walmart stores using various machine learning models.
- **Model Comparison**: Compare the performance of different models (Linear Regression, Random Forest, KNN).
- **Interactive Dashboard**: Visualize sales data and predictions through an interactive dashboard.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/walmart-sales-forecasting.git
    cd walmart-sales-forecasting
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the Flask application:
    ```bash
    export FLASK_APP=app.py
    export FLASK_ENV=development
    ```

5. Run the application:
    ```bash
    flask run
    ```

## Usage
1. Navigate to `http://127.0.0.1:5000` in your web browser.
2. Upload the sales data file.
3. Select the machine learning model to use for forecasting.
4. View the forecasted sales and model performance metrics.

## Models
- **Linear Regression**: A simple linear approach to modeling the relationship between the dependent and independent variables.
- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees.
- **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification and regression.

## Data
The dataset used for this project is from the Walmart Recruiting - Store Sales Forecasting competition on Kaggle. It includes historical sales data for 45 Walmart stores located in different regions.

## Contributing
We welcome contributions! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Kaggle for providing the dataset.
- The Flask community for their excellent documentation and support.

## Contact
For any questions or suggestions, please contact us at [your-email@example.com].
