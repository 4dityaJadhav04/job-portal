import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('titanic')
df.head()

df = df[['survived','pclass','sex','age','fare']].dropna()

#label encoding 
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

#split the data 
X = df.drop('survived' , axis = 1)
y = df['survived']
X_train, X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

#model 
model = LogisticRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print(f"Accuracy: {accuracy*100:.2f}")


survived = y_test.sum()
not_survived = len(y_test) - survived

plt.bar(["Survived" , "Not Survived"] , [survived , not_survived])
plt.title("Survival Distribution")
plt.ylabel("Number of Passengers")
plt.show()