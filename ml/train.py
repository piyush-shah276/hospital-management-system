"""Trains LR / Decision Tree / Random Forest per disease, keeps the best, saves it.
Real data: put Pima 'diabetes.csv' and UCI 'heart.csv' inside /data and re-run this file.
If a CSV is missing, a synthetic DEMO dataset with the same columns is used."""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); M = os.path.join(H, 'models')
FEATS = {'diabetes': [('Pregnancies','Pregnancies'),('Glucose','Glucose (mg/dL)'),('BloodPressure','Blood Pressure (mm Hg)'),('BMI','BMI'),('Age','Age')],
         'heart': [('age','Age'),('sex','Sex (1=M, 0=F)'),('cp','Chest pain type (0-3)'),('trestbps','Resting BP'),('chol','Cholesterol'),('thalach','Max heart rate'),('exang','Exercise angina (1/0)'),('oldpeak','ST depression')],
         'liver': [('Age','Age'),('Total_Bilirubin','Total Bilirubin (mg/dL)'),('ALT','ALT / SGPT (U/L)'),('AST','AST / SGOT (U/L)'),('Albumin','Albumin (g/dL)')],
         'hypertension': [('age','Age'),('BMI','BMI'),('Systolic_BP','Systolic BP (mm Hg)'),('Diastolic_BP','Diastolic BP (mm Hg)'),('Cholesterol','Cholesterol (mg/dL)'),('Smoker','Smoker (1=Yes, 0=No)')]}
TARGET = {'diabetes': 'Outcome', 'heart': 'target', 'liver': 'target', 'hypertension': 'target'}

def synth(d, n=1500):
    r = np.random.default_rng(42)
    if d == 'diabetes':
        X = pd.DataFrame({'Pregnancies': r.integers(0,14,n), 'Glucose': r.normal(120,30,n).clip(50,200), 'BloodPressure': r.normal(72,12,n).clip(40,120), 'BMI': r.normal(32,7,n).clip(18,60), 'Age': r.integers(21,80,n)})
        z = .035*(X.Glucose-120)+.06*(X.BMI-32)+.03*(X.Age-40)+.08*X.Pregnancies-.5
    elif d == 'heart':
        X = pd.DataFrame({'age': r.integers(30,78,n), 'sex': r.integers(0,2,n), 'cp': r.integers(0,4,n), 'trestbps': r.normal(130,17,n).clip(90,200), 'chol': r.normal(245,50,n).clip(120,500), 'thalach': r.normal(150,22,n).clip(70,205), 'exang': r.integers(0,2,n), 'oldpeak': r.gamma(1.2,.9,n).clip(0,6)})
        z = .05*(X.age-54)+.01*(X.trestbps-130)+.006*(X.chol-245)-.03*(X.thalach-150)+.9*X.exang+.8*X.oldpeak+.4*(X.cp==0)-.6
    elif d == 'liver':
        X = pd.DataFrame({'Age': r.integers(20,80,n), 'Total_Bilirubin': r.gamma(1.5,.8,n).clip(.2,15), 'ALT': r.gamma(2,25,n).clip(10,400), 'AST': r.gamma(2,28,n).clip(10,400), 'Albumin': r.normal(3.8,.6,n).clip(1.5,5.5)})
        z = .5*(X.Total_Bilirubin-1.5)+.01*(X.ALT-50)+.008*(X.AST-55)-.8*(X.Albumin-3.8)-.3
    else:
        X = pd.DataFrame({'age': r.integers(25,80,n), 'BMI': r.normal(27,5,n).clip(16,50), 'Systolic_BP': r.normal(125,18,n).clip(90,200), 'Diastolic_BP': r.normal(80,11,n).clip(55,120), 'Cholesterol': r.normal(200,35,n).clip(120,320), 'Smoker': r.integers(0,2,n)})
        z = .04*(X.age-45)+.08*(X.BMI-27)+.05*(X.Systolic_BP-130)+.05*(X.Diastolic_BP-85)+.01*(X.Cholesterol-200)+.5*X.Smoker-.3
    X[TARGET[d]] = (r.random(n) < 1/(1+np.exp(-z))).astype(int)
    return X

def load(d):
    cols = [c for c,_ in FEATS[d]]; p = os.path.join(D, d+'.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)[cols+[TARGET[d]]]
        if d == 'diabetes':
            for c in ['Glucose','BloodPressure','BMI']: df[c] = df[c].replace(0, df[c].median())
        return df.dropna(), 'real dataset'
    return synth(d), 'demo (synthetic) data'

def train_all():
    os.makedirs(M, exist_ok=True); info = {}
    for d in FEATS:
        df, src = load(d); X = df[[c for c,_ in FEATS[d]]]; y = df[TARGET[d]]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1, stratify=y)
        cands = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=1), 'Random Forest': RandomForestClassifier(200, random_state=1)}
        res = {}
        for n, m in cands.items():
            pipe = make_pipeline(StandardScaler(), m).fit(Xtr, ytr); pr = pipe.predict(Xte)
            res[n] = (accuracy_score(yte,pr), precision_score(yte,pr,zero_division=0), recall_score(yte,pr), pipe)
        best = max(res, key=lambda k: res[k][0]); joblib.dump(res[best][3], os.path.join(M, d+'.joblib'))
        info[d] = {'source': src, 'best': best, 'models': {k: [round(v[0]*100,1), round(v[1]*100,1), round(v[2]*100,1)] for k,v in res.items()}}
    json.dump(info, open(os.path.join(M, 'metrics.json'), 'w'), indent=1); return info

if __name__ == '__main__': print(json.dumps(train_all(), indent=1))
