from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

iris = load_iris()
X = iris.data
y = iris.target 
target_names = iris.target_names

X_train , X_test ,  y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled , y_train)

y_pred = model.predict(X_test_scaled)
accuracy_score = accuracy_score(y_test , y_pred)
print(f"Accuracy of the Model : {accuracy_score*100:.2f}%")

joblib.dump(model , 'iris_logistic_model.pkl')
joblib.dump(scaler , 'scaler.pkl')
print("Model and scaler saved Successfully ")

#reuse the model 
loaded_model = joblib.load('iris_logistic_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')


#Make Prediction on new data : 
new_data = [[5.1 , 3.5 , 1.4 , 0.2]]
new_data_scaled = loaded_scaler.transform(new_data)
new_prediction = loaded_model.predict(new_data_scaled)

predicted_class_name = target_names[new_prediction[0]]
print(f"Prediction for new data : {predicted_class_name}")
