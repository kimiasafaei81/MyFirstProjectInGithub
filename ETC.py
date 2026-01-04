import seaborn as sns
import matplotlib.pyplot as plt

X = [10,20,30,40,50]
# print(max(X))
# m = max(X)
# n = min(X)
# print(m , n)
# X_Normal = []
# for item in X:
#     X_Normal.append((item-n) / (m - n))
# X_Normal2 = [(item-n) / (m - n) for item in X]
# print(X_Normal2)
# X_2D=[[]]
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# model = MinMaxScaler()
# X_2D = np.array(X).reshape(-1,1)
# X_norm = model.fit_transform(X_2D)
# print(X_norm)
#
#
# x = np.array(X)
# print(x)
# from sklearn.preprocessing import StandardScaler
# model = StandardScaler()
# x_standars = model.fit_transform(x.reshape(-1,1))
# print(x_standars)
#
# mean = np.mean(x)
# std = np.std(x)
# x_Standard = (X-mean) / std
# print(x_Standard)
# y = [10,20,30,40,50,60]
# q = np.array(y).reshape(-1,2)
# print(q)
import pandas as pd
from scipy.special.cython_special import huber

df = pd.read_csv('titanic.csv')
male_passengers = df.query("Sex == 'male'")
print(male_passengers['Sex'].unique())
print(male_passengers.to_string())
# print(df.columns)
# print(df.info())
# print(df.describe().to_string())
# print(df.isnull().sum())
# print(df.isna())
# print(len(df))
# print(df.shape)
print(df['Sex'].unique())
print(df['Embarked'].unique())
avg_age = df['Age'].mean()
median_Age = df['Age'].median()
mod_Age = df['Age'].mode()[0]
print(avg_age, median_Age , mod_Age)
df['Age'] = df['Age'].fillna(avg_age)
df = df.drop('Cabin',axis = 1)
print(df.columns)
mode_Embarked = df['Embarked'].mode()[0]
# print(mode_Embarked)
df['Embarked'] = df['Embarked'].fillna(mode_Embarked)
# print(df.isnull().sum())
# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)

print(df.dtypes )
# print(df.corr(numeric_only=True).to_string( ))
print(df['Sex'].unique())
print(df['Embarked'].unique())
df['Embarked'] = df['Embarked'].map({'S':1, 'C':2, 'Q':3})
print(df['Embarked'].unique())
df['Sex'] = df['Sex'].map({'male':1, 'female':2})
print(df['Sex'].unique())
df = df.drop(['Name','Ticket'],axis=1)
print(df.to_string())
print(df['Age'].mean())
print(df['Age'].skew())
# sns.histplot(df,x='Age')
# plt.title('histplot')
# plt.show()
# ax = sns.barplot(data=df,x='Sex', y='Survived',hue='Pclass')
# ax.set_xticklabels(['Male','Female'])
# plt.title('barplot')
# # sns.palplot(df)
# plt.show()
# # sns.histplot(df,x='Age',bins = 30,kde=True)
# ax = sns.countplot(df,x= 'Sex',hue='Survived')
# for p in ax.patches:
#     height = p.get_height()
#     formated = f"{height:,.1f}"
#     ax.text(
#         p.get_x() + p.get_width() /2, height,formated,ha='center', va = 'bottom',fontsize = 10,color = 'red')
# ax.set_xticklabels(['Male' , 'Female'])
# plt.title('Survived Count by Sex')
# plt.show()
ax = sns.barplot(df,x = 'Pclass',y = 'Survived')
for p in ax.patches:
    h= 0.07
    height = p.get_height()
    formated = f"{(height)*100:.1f}%"
    ax.text(p.get_x()+p.get_width()/2 , height/2 , formated
            , ha = 'center' , va= 'bottom' , color = 'blue', fontsize = 10)
ax .set_xticklabels(['P1' , 'P2', 'P3'])
plt.show()
# ax = sns.boxplot(df,x = 'Pclass' , y= 'Fare' )
# plt.show()
# ax = sns.kdeplot(df,x = 'Age' , hue = 'Survived')
# plt.show()
# ax = sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
# plt.show()
# ax = sns.pairplot(df,hue = 'Survived')
# plt.show()
ax = sns.catplot(df,x = 'Embarked',y = 'Survived' , kind = 'point' , hue = 'Sex')
plt.show()
ax = sns.scatterplot(df,x='Fare',y='Age',hue = 'Survived')
plt.show()