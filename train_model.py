import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
X = data.drop('Outcome', axis =1)
y = data['Outcome']
model = LogisticRegression()

model.fit(X, y)
joblib.dump(model, 'models\\logreg_model.joblib')
