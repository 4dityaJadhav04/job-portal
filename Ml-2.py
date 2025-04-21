import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os 

iris = load_iris()
X = iris.data 
y = iris.target 

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# Model Training - Version 1 (Default Model)
model_v1 = RandomForestClassifier(random_state=42)
model_v1.fit(X_train , y_train)
y_pred_v1 = model_v1.predict(X_test)
acc_v1 = accuracy_score(y_test , y_pred_v1)
print(f"Model v1 Accuracy : {acc_v1*100:.2f}%")

# Hyperparameter Tunning - Version 2 
param_grid = {
    'n_estimators':[50,100],
    'max_depth':[2,4,6]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42) , param_grid , cv=3 )
grid_search.fit(X_train, y_train)

model_v2 = grid_search.best_estimator_
y_pred_v2 = model_v2.predict(X_test)
acc_v2 = accuracy_score(y_test , y_pred_v2)
print(f"Model v2 Accuracy : {acc_v2*100:.2f}%")

results = pd.DataFrame({
    'version':['v1' , 'v2'],
    'Accuracy':[acc_v1 , acc_v2]
})

print("Model Comarison : \n",results)

if not os.path.exists("models") :
    os.makedirs("models")

joblib.dump(model_v1 , "models/model_v1.pkl")
joblib.dump(model_v2 , "models/model_v2.pkl")
print("\n Models saved as 'model_v1.pkl' and 'model_v2.pkl' in 'models/' folder.")