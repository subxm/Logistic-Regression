# Telecom Churn Predictor AI

A modern, dark-themed Streamlit web application for predicting customer churn in a telecom company using Logistic Regression.

## Overview

This application analyzes customer data to predict whether a customer is likely to churn (leave the service). It uses a Logistic Regression model trained on historical customer data including tenure, monthly charges, and total charges.

## Features

- **Churn Prediction** - Enter customer details and get instant churn probability predictions
- **Interactive Visualizations** - Beautiful gauge charts showing prediction confidence
- **Model Analytics** - View model performance metrics including accuracy, confusion matrix, and classification reports
- **Dataset Explorer** - Browse the dataset with interactive charts showing churn distribution, tenure patterns, and charge distributions

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run streamlit_app.py
```

## Project Structure

```
telecom-churn-predictor/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── telecom_churn_cleaned.csv # Dataset
└── README.md                 # This file
```

## Dataset

The dataset contains 7,049 telecom customers with the following features:

| Feature | Description |
|---------|-------------|
| Tenure | Customer relationship duration in months |
| Monthly Charges | Monthly service cost in USD |
| Total Charges | Cumulative service cost in USD |
| Churn | Whether customer left (Yes/No) |

## Model Details

- **Algorithm**: Logistic Regression
- **Training Samples**: 5,639
- **Test Samples**: 1,410
- **Accuracy**: ~75%

## Screenshots

### Prediction Page
Enter customer details to predict churn probability with interactive gauge visualization.

### Analytics Page
View model performance metrics, confusion matrix, and prediction distribution charts.

### Dataset Explorer
Explore the dataset with summary statistics and interactive visualizations.

## Deployment

This app is ready to deploy on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set the main file to `streamlit_app.py`
4. Deploy!

## License

MIT License

## Author

Subham Singh Negi