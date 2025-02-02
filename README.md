# Autism_Prediction_Using_MachineLearning

**Project Overview**

This project focuses on predicting Autism Spectrum Disorder (ASD) using machine learning models. The dataset consists of various attributes related to individuals, and the goal is to build a predictive model to identify cases of autism.

**Technologies Used**

Python

Pandas

NumPy

Scikit-learn

XGBoost

Seaborn

Matplotlib

Imbalanced-learn (SMOTE)

Pickle

---

**Dataset**

The dataset contains multiple features related to an individual's characteristics and behavioral traits. The data undergoes preprocessing before feeding it into the machine learning models.

---

**Steps Involved**

**1. Data Loading & Preprocessing**

Load the dataset from a CSV file.

Handle missing values.

Convert categorical features to numerical using Label Encoding.

Balance the dataset using SMOTE to address class imbalance.

**2. Model Training**

Split the dataset into training and testing sets.

Train multiple classification models:

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Perform hyperparameter tuning using RandomizedSearchCV.

**3. Model Evaluation**

Evaluate models based on:

Accuracy Score

Confusion Matrix

Classification Report

**4. Model Deployment**

Save the trained model using Pickle for future use.

---

**Installation & Usage**

1.Clone the repository:

git clone https://github.com/yourusername/Autism_Prediction.git

cd Autism_Prediction

2.Install required dependencies:

pip install -r requirements.txt

3.Run the Jupyter Notebook to train and evaluate the model.

4.Load the saved model and use it for predictions.

---

**Results**

The trained model helps in predicting autism with high accuracy. The best model is selected based on evaluation metrics.
