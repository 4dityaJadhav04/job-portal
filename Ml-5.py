import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from ydata_profiling import ProfileReport

df = sns.load_dataset('titanic')
df.head()

print("Shape of the Dataset:",df.shape)
print("\nInfo About the Dataset:")
df.info()

print("\nMissing Values:\n",df.isnull().sum())

# Handling Null values 
df['age'].fillna(df['age'].median() , inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0] , inplace=True)
df.drop(columns='deck' , inplace=True)

print("\nMissing Values After Cleaning :\n",df.isnull().sum())

print("\nSummary Statistics:\n",df.describe())

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True) , annot=True , cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

#Age distribution 
plt.figure(figsize=(8, 5))
sns.histplot(df['age'] , kde=True , bins=30 , color='teal')
plt.title("Age Distribution")
plt.show()

# Survival by Gender 
plt.figure(figsize=(6,4))
sns.countplot(data = df , x='sex' , hue='survived',palette='Set2')
plt.title('Survival Count by Gender')
plt.show()

# Embark Town vs Survival
plt.figure(figsize=(6, 4))
sns.countplot(data=df , x='embark_town' , hue="survived" , palette='Set1')
plt.title('Survival by Embark Town')
plt.show()

# # Generate Profile Report (Optional)
# profile = ProfileReport(df, title="Titanic Data EDA Report", explorative=True)
# profile.to_file("titanic_eda_report.html")

# print("\n✅ EDA completed. Report saved as 'titanic_eda_report.html'")

# jupyter nbconvert --to pdf EDA_Titanic.ipynb

