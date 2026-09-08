# 🏠 House Price Prediction API

A Machine Learning-based REST API for predicting California house prices using a Random Forest Regression model.

This project is built using FastAPI and Scikit-learn. It supports both single house predictions and batch predictions using CSV file uploads.

---

## 🚀 Features

- 🏠 Single house price prediction
- 📁 Batch prediction using CSV file upload
- 🤖 Random Forest Regression model
- ✅ Input validation using Pydantic
- 🔍 CSV validation for missing values
- 🔢 Non-numeric value validation
- 📍 Latitude and Longitude range validation
- ❤️ Health check endpoint
- 📖 Automatic API documentation using Swagger UI
- 📥 Download prediction results as a CSV file

---

## 🛠️ Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pydantic
- Uvicorn

---

## 📂 Project Structure

```text
House_prediction_api/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py
├── train.py
├── explore.py
│
├── random_forest_model.joblib
└── model_columns.joblib
