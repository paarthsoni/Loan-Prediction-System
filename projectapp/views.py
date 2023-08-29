from django.shortcuts import render, HttpResponse, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import numpy as np
from .models import *


def index(request):
    return render(request, 'index.html')


def loan_prediction(request):

    gender = request.POST.get('gender')
    marital_status = request.POST.get('maritalStatus')
    dependents = request.POST.get('dependents')
    education = request.POST.get('education')
    self_employed = request.POST.get('selfEmployed')
    applicantincome = request.POST.get('applicantIncome')
    coapplicantincome = request.POST.get('coapplicantIncome')
    print(gender, marital_status, dependents, education,
          self_employed, applicantincome, coapplicantincome)
    if gender == 'Male':
        gendercode = 0
    elif gender == 'Female':
        gendercode = 1
    else:
        gendercode = 2

    if dependents == 0:
        dependentcode = 0
    elif dependents == 1:
        dependentcode = 1
    elif dependents == 2:
        dependentcode = 2
    else:
        dependentcode = 3

    if marital_status == 'Yes':
        marital_status_code = 1
    else:
        marital_status_code = 0

    if education == 'Graduate':
        educationcode = 1
    else:
        educationcode = 0

    if self_employed == 'Yes':
        self_employed_code = 1
    else:
        self_employed_code = 0

    # Load your dataset into a Pandas DataFrame
    df = pd.read_csv('/home/paarth1234/ML_internship_inhouse_project/dataset/Loan_Data.csv')
    # Separate features and target
    X = df[['ApplicantIncome', 'CoapplicantIncome']]
    y = df['Loan_Status']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Now, you can use this model for loan predictions
    # For example, you can input user data:
    # applicant = 50
    # coapplicant = 100
    # Input values based on column order and encoding
    user_data = [applicantincome, coapplicantincome]
    predicted_loan_status = model.predict([user_data])[0]
    print(f'Predicted Loan Status: {predicted_loan_status}')

    if predicted_loan_status == 'N':
        userdata = loanpredictiondata(gender=gender, marital_status=marital_status, dependents=dependents, education=education,
                                      self_employed=self_employed, applicantincome=applicantincome, coapplicantincome=coapplicantincome, loan_status=predicted_loan_status, loan_amount=0)

        userdata.save()
        return render(request, 'loan_prediction.html', {'loan_status': 'Not Eligible'})
    if predicted_loan_status == 'Y':
        # Load your dataset into a Pandas DataFrame
        df = pd.read_csv('/home/paarth1234/ML_internship_inhouse_project/dataset/Loan_Data.csv')

        # Data preprocessing
        # Drop Loan_ID as it's not relevant for prediction
        df = df.drop('Loan_ID', axis=1)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
            df.select_dtypes(include=[np.number]))

        # Encode categorical features
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Separate features and target
        X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome']]
        y = df['LoanAmount']

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Initialize and train the Gradient Boosting model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.2f}')

        # Now, you can use this trained model to predict loan amounts
        # For example, you can input user data similar to this:
        # Input values based on column order and encoding
        user_data = [gendercode, marital_status_code, dependentcode, educationcode,
                     self_employed_code, applicantincome, coapplicantincome]
        predicted_loan_amount = model.predict([user_data])
        predicted_loan_amount = float(predicted_loan_amount)
        print(round(predicted_loan_amount, 2)*1000)
        userdata = loanpredictiondata(gender=gender, marital_status=marital_status, dependents=dependents, education=education,
                                      self_employed=self_employed, applicantincome=applicantincome, coapplicantincome=coapplicantincome, loan_status=predicted_loan_status, loan_amount=round(predicted_loan_amount, 2)*1000)
        userdata.save()
        return render(request, 'loan_prediction.html', {'loan_status': 'Eligible', 'loan_amount': round(predicted_loan_amount, 2)*1000})
