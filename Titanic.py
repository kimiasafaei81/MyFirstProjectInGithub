import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
df = pd.read_csv('titanic.csv')
avg_age = df['Age'].mean()
median_avg = df['Age'].median()
mode_embarked = df['Embarked'].mode()[0]
df = df.drop(['Name','Cabin','Ticket'],axis=1)
df['Age']=df['Age'].fillna(avg_age)
df['Embarked']=df['Embarked'].fillna(mode_embarked)
df['Sex']=df['Sex'].map({'male':1,'female':2})
df['Embarked']=df['Embarked'].map({'S':1,"C":2,'Q':3})
sns.histplot(df,x='Fare')
plt.show()
sns.histplot(df,x='Age')
plt.show()
sns.boxplot(df,x='Fare')
plt.grid(axis='x' , alpha = 0.3)
plt.show()
plt.figure(figsize=(10,6))
sns.boxplot(df,x ='Age')
plt.grid(axis = 'x' , alpha = 0.3)
plt.show()
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5 * IQR
print('Lower_Bound = ' , lower_bound)
print('Upper Bound = ',upper_bound)

remove_outlier_data = df[(df['Fare']< lower_bound) | (df['Fare'] > upper_bound)]
cleanDF = df[(df['Fare']>= lower_bound) & (df['Fare']<= upper_bound)]
print(df.shape)
print(remove_outlier_data.shape)
fig , axis = plt.subplots(nrows = 1 , ncols = 3, figsize = (14 , 6))
sns.violinplot(df,x='Fare',ax=axis[0])
axis[0].set_title("DataFrame Original")
# axis[0].axis("off")
sns.violinplot(remove_outlier_data,x='Fare',ax = axis[1])
axis[1].set_title("DataFrame Remove ")
# axis[1].axis("off")
sns.violinplot(cleanDF,x='Fare',ax = axis[2])
axis[2].set_title("Final DataFrame")
plt.tight_layout()
plt.show()