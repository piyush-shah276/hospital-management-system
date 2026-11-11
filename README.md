 # MediCare HMDPS – Hospital Management & Disease Prediction System

A full-stack hospital management web app with machine-learning based disease risk screening, built with Flask and SQLite.

> **Disclaimer:** The disease prediction feature is for risk screening and academic demonstration only. It is not a substitute for professional medical diagnosis.
 

## Features

- Manage patients, doctors, appointments, medicines, billing and staff records
- Role-based access for
- Online appointment booking with token numbers and email confirmation
- Doctors can view only their assigned patients and specialty-related records
- ML risk screening for diabetes, heart disease, liver disease and hypertension (Logistic Regression, Decision Tree, Random Forest)
- PDF invoices and medical reports generated with ReportLab
- Installable on mobile as a Progressive Web App (PWA)

## Tech Stack

Python, Flask, SQLite, scikit-learn, Pandas, NumPy, HTML, CSS, JavaScript, Bootstrap, ReportLab

## Project Structure

```
ml/          Model training code and saved models
database/    SQLite database
data/        Datasets
templates/   HTML pages (UI)
static/      CSS and JavaScript
```

## How to Run

1. Install Python 3.9 or later.
2. Clone the repository:
   ```
   git clone https://github.com/piyush-shah276/<repo-name>.git
   cd <repo-name>
   ```
3. Start the app:
   - **Windows:** double-click `run.bat`
   - **Linux / Mac:** `bash run.sh`
4. Open http://127.0.0.1:5000 in your browser.
 

## Datasets and Models

The models are trained on public datasets (add the exact source links here, for example the Pima Indians Diabetes dataset and the UCI Heart Disease dataset from Kaggle).

To retrain with your own copy of the data, put the CSV files in the `data/` folder, delete `ml/models`, and run the app again. The models retrain automatically.

| Disease | Best model | Accuracy |
|---------|-----------|----------|
| Diabetes | <add> | <add> |
| Heart disease | <add> | <add> |
| Liver disease | <add> | <add> |
| Hypertension | <add> | <add> |

## Author

**Piyush Kumar** – BCA (AI & Data Science), Graphic Era Deemed to be University, Dehradun
[LinkedIn](https://www.linkedin.com/in/piyush-kumar-09a056445) | [GitHub](https://github.com/piyush-shah276)
