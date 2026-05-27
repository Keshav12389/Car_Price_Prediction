# 🚗 Car Price Prediction — ANN Project

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

## 📌 Project Overview

This project predicts the **resale price of used cars** in the Indian market using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras. The model is trained on 301 real used car listings and evaluated using R² score to measure prediction accuracy.

The project covers the full data science pipeline — from raw data loading and EDA, through feature engineering and preprocessing, to model building and evaluation.

**Dataset:** [Vehicle Dataset](https://github.com/Keshav12389/Car_Price_Prediction/blob/main/car%20data.csv)  
**Total Records:** 301 used car listings  
**Tool Used:** Python, Jupyter Notebook  
**Author:** Keshav Meena | IIT Delhi

---

## 🎯 Problem Statement

Used car prices depend on many factors — age, mileage, fuel type, transmission, and seller type. The goal is to build a model that can **accurately predict a car's resale price** given these features, helping buyers and sellers make informed decisions.

---

## 🗂️ Dataset Description

| Column | Description |
|---|---|
| `Car_Name` | Name of the car model |
| `Year` | Year of manufacture |
| `Selling_Price` | Price at which car is being sold (in Lacs) |
| `Present_Price` | Current ex-showroom price (in Lacs) |
| `Kms_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

---

## 🔁 Project Pipeline

```
Raw Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Preprocessing & Feature Engineering
    ↓
Train-Test Split (80/20)
    ↓
Feature Scaling (StandardScaler)
    ↓
ANN Model Building (TensorFlow/Keras)
    ↓
Model Training (100 Epochs)
    ↓
Evaluation (R² Score)
```

---

## 🛠️ Steps Performed

### 1. Exploratory Data Analysis
- Checked dataset shape, data types, and null values
- Visualized distribution of categorical features: Fuel Type, Seller Type, Transmission, Past Owners using countplots
- Detected outliers using 99th percentile threshold for Present Price, Selling Price, and Kms Driven

### 2. Feature Engineering
- Created new feature **`Age`** = 2021 - Year (dropped original `Year` column)
- Renamed columns for clarity: `Selling_Price` → `Selling_Price(lacs)`, `Owner` → `Past_Owners`
- Dropped `Car_Name` column (non-numeric, high cardinality)

### 3. Data Preprocessing
- Applied **One-Hot Encoding** using `pd.get_dummies()` on 3 categorical columns:
  - `Fuel_Type` (Petrol/Diesel/CNG)
  - `Seller_Type` (Dealer/Individual)
  - `Transmission` (Manual/Automatic)
- Generated **correlation heatmap** to identify top predictors
- Built **pivot table** — Seller Type vs Fuel Type by average Selling Price

### 4. Train-Test Split
- Split: **80% train / 20% test** (`test_size=0.2`, `random_state=1`)
- Train set: 240 records | Test set: 61 records

### 5. Feature Scaling
- Applied **StandardScaler** on all features
- Fit on train set, transformed both train and test sets (no data leakage)

### 6. ANN Architecture

```
Input Layer  →  30 neurons (ReLU)
                      ↓
Hidden Layer →  10 neurons (ReLU)
                      ↓
Output Layer →   1 neuron  (Linear — regression output)
```

```python
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')
```

### 7. Model Training
- **Epochs:** 100
- **Optimizer:** RMSprop
- **Loss Function:** Mean Squared Error (MSE)
- Tracked training and validation loss across all epochs

### 8. Model Evaluation
- Evaluated using **R² Score** on both train and test sets
- Plotted training vs validation loss curve to check for overfitting

---

## 📊 Results

| Metric | Value |
|---|---|
| R² Score — Train | [ADD YOUR VALUE] |
| R² Score — Test | [ADD YOUR VALUE] |
| Difference (Train - Test) | [ADD YOUR VALUE] |

> **Note:** Open the notebook and check the printed output of the last 3 cells to find your exact scores and fill them in above.

---

## 💡 Key Findings

- **Present Price** and **Car Age** are the strongest predictors of resale value (highest correlation with Selling Price)
- **Dealer-sold cars** have significantly higher average prices than individually sold cars across all fuel types
- **Diesel cars** command higher resale prices than Petrol cars on average
- **Automatic transmission** cars are priced higher but represent only ~15% of the dataset
- Cars with **0 previous owners** are priced considerably higher than those with 1 or more owners

---

## 📁 Repository Structure

```
car-price-prediction/
│
├── README.md                        ← Project overview
├── car_price_prediction_ann.ipynb   ← Main Jupyter Notebook
└── dataset/
    └── car data.csv                 ← Source dataset (from Kaggle)
```

---

## 🚀 How to Run This Project

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Keshav12389/car-price-prediction.git
cd car-price-prediction
```

**Step 2 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

**Step 3 — Add the dataset**
- Download `car data.csv` from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Place it in the `dataset/` folder

**Step 4 — Run the notebook**
```bash
jupyter notebook car_price_prediction_ann.ipynb
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Pandas | Data loading, manipulation, feature engineering |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization, countplots, heatmap |
| Scikit-learn | Train-test split, StandardScaler, R² score |
| TensorFlow / Keras | ANN model building and training |

---

## 📊 Sample Code — ANN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

# Build model
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')

# Train
model.fit(X_train, y_train, epochs=100,
          validation_data=(X_test, y_test))

# Evaluate
test_pred = model.predict(X_test)
print("R² Test:", r2_score(y_test, test_pred))
```

---

## 🔗 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/keshav-meena)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Keshav12389)

---

*This project is part of my data analytics and machine learning portfolio.*
