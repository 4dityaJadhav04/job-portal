import os 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay , confusion_matrix , 
    precision_recall_curve , average_precision_score ,
    classification_report
)

os.makedirs("results",exist_ok=True)

data = load_breast_cancer()
X   = data.data  
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train models 
model1 = LogisticRegression(max_iter=10000)
model2 = DecisionTreeClassifier()

model1.fit(X_train , y_train)
model2.fit(X_train , y_train)

# predictions 
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

proba1 = model1.predict_proba(X_test)[: ,1]
proba2 = model2.predict_proba(X_test)[: ,1]


# Confusion Matrix 
for name , pred in zip(['logistic','tree'],[pred1 , pred2]) : 
    cm = confusion_matrix(y_test , pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{name} - Confusion Matrix ')
    plt.savefig(f'results/{name}_confusion_matrix.png')
    plt.close()

# Precison-Recall Curve
for name , proba in zip(['Logistic Regression','Decision Tree'] , [proba1,proba2]) : 
    precison , recall , _ = precision_recall_curve(y_test , proba)
    ap = average_precision_score(y_test , proba)
    plt.plot(recall , precison , label=f'{name} (AP={ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('results/precision_recall_comparison.png')
plt.close()

 # Print comparison
print("Logistic Regression:\n", classification_report(y_test, pred1))
print("Decision Tree:\n", classification_report(y_test, pred2))