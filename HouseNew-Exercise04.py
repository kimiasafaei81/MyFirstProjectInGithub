import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno

# from mm import y_pridict

df = pd.read_csv('HouseNew.csv',usecols=['Elevator','Floor','Area','Parking','Room',
                                         'Price','Warehouse','YearOfConstruction','Address'])
# print(df)
# print('--------------------- shape ------------)')
# print(df.shape)
# print('--------------------- columns ------------)')
# print(df.columns) #Tuple
# print('--------------------- dtypes ------------)')
# print(df.dtypes)
# print('--------------------- describe() ------------)')
# print(df.describe())
# print('--------------------- info ------------)')
# print(df.info())
# print('--------------------- isna().sum() ------------)')
# print(df.isnull().sum())
# print(df.head(10).to_string())
# print(df.tail(15).to_string())
# missingno.matrix(df)
# plt.show()
df["Age"] = 1404 - df['YearOfConstruction']
df["PricePerM2"] = round(df["Price"] / df["Area"],2)
# print(df['Elevator'])
# df['Elevator'] = df['Elevator'].map({True:1,False:0})
df['Elevator'] = df['Elevator'].astype(int)
df['Parking'] = df['Parking'].map({True:1,False : 0}) #dic : { 'oldValue': 'New Value'}
df['Warehouse'] = df['Warehouse'].map({True:1,False:0})
# print(df['Elevator'])
# df['HasElevatorParking'] = df['Elevator']*1 + df['Parking']*1 + df['Warehouse']*1
df['HasElevatorParking'] = df['Elevator'] + df['Parking'] + df['Warehouse']
# print(df.dtypes)
df['Address'] = df['Address'].fillna('نامشخص')
addressCount = df['Address'].value_counts()
# print(addressCount.dtypes)
# print(addressCount.tail(12).to_string())

# print(addressCount)
df['AddressFrequency'] = (df['Address'].map(addressCount))
# print(df.columns)
# print(df[['Price','YearOfConstruction', 'Age' , 'PricePerM2','HasElevatorParking','AddressFrequency']].tail(12).to_string())
print(df['Floor'].unique())
median_Floor = df['Floor'].median()
print(median_Floor)
df['Floor'] = df['Floor'].fillna(median_Floor).astype(int)
print(df.info())
# print(df.corr(numeric_only=True).to_string())
# print(df.head(10).to_string())
# print(df['Address'].nunique())
# print(df['Address'].unique())
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
le = LabelEncoder()
df['AddressEncoded'] = le.fit_transform(df['Address'])
# print('------------------------------')
# print(df.columns)
# print(df[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Price', 'Warehouse',
#        'YearOfConstruction',  'Age', 'PricePerM2',
#        'HasElevatorParking', 'AddressFrequency', 'AddressEncoded']] .head(10).to_string())
# print(df.info())
# print(df[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Price', 'Warehouse',
#        'YearOfConstruction',  'Age', 'PricePerM2',
#        'HasElevatorParking', 'AddressFrequency', 'AddressEncoded']][df['AddressEncoded']==0].to_string())
# X = df.drop('Price',axis=1)
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
X = df[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Warehouse',
         'Age', 'PricePerM2',
       'HasElevatorParking', 'AddressEncoded']]
# X = df[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Warehouse',
#          'Age', 'PricePerM2',
#        'HasElevatorParking', 'AddressFrequency', 'AddressEncoded']]
# X = df.iloc[: , :] # روش سوم تولید ایکس
# print(X.to_string())
y = df['Price']
# print(y)
ss = StandardScaler()
X_Scaled = ss.fit_transform(X)
# print(X_Scaled )
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_Scaled,y,test_size=0.3,random_state=42)
# print(y_train)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pridict = lr.predict(X_test)
sns.heatmap(X.corr(numeric_only=True),annot=True,cmap='rainbow')
plt.show()

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
mae = f"{mean_absolute_error(y_true=y_test,y_pred=y_pridict):,.0f}"
mse = f"{mean_squared_error(y_true=y_test,y_pred=y_pridict):,.0f}"
rmse = f"{np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_pridict)):,.0f}"
r2 = f"{r2_score(y_true=y_test,y_pred=y_pridict):.2%}"
print(f"MAE:{mae}")
print(f"MSE = {mse}")
print(f"RMSE = {rmse}")
print(f"R2 Score ={r2}")
# sns.pairplot(df)
# plt.show()

