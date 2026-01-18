# Social-Media-Emotion-Classifier [Link:]([https://i2lzqk8wrgbecnzzv99laj.streamlit.app](https://social-media-emotion-classifier-nsd6uz4lppfpfme2tzz995.streamlit.app/))


## 📌 Overview
This project implements an end-to-end machine learning classification pipeline using Python. It includes data loading, preprocessing, model training, and performance evaluation using multiple classification algorithms from scikit-learn.

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## 📊 Dataset
- **train.csv** – used for model training  
- **test.csv** – used for testing and predictions  

Data is loaded using `pd.read_csv()` and explored using `head()`, `info()`, and `describe()`.

## 🧹 Data Preprocessing
- Missing values handled using `dropna()`  
- Datasets combined where required for consistent preprocessing  
- Features and target variable separated  
- Data split using `train_test_split()`  

## 🤖 Models Implemented
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Bagging Classifier  
- AdaBoost Classifier  

Models are trained using the `fit()` method and predictions are generated using `predict()`.

## 📈 Model Evaluation
- Accuracy Score used as the evaluation metric  
- Model performances compared to identify the best model  
- Visualizations created using Matplotlib and Seaborn  
