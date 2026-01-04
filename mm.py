import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('HouseNew.csv',usecols=['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Price', 'Warehouse',
       'YearOfConstruction', 'Address'])
df['Address'] = df['Address'].fillna('نامشخص')
# print(df['Floor'].unique())
mode_floor = df['Floor'].mode()[0]
# print(mode_floor)
df['Floor']= df['Floor'].fillna(mode_floor)
# print(df.isnull().sum())
# address_count = df['Address'].value_counts()
# df['Address_frequency'] = df['Address'].map(address_count)
# print(df['Address_frequency'])
df['Age'] = 1404 - df['YearOfConstruction']
le = LabelEncoder()
df['Address_Encoded'] = le.fit_transform(df['Address'])
# print(df['Address_Encoder'])
# print(df['Parking'])
df['Elevator'] = df['Elevator'].astype('int64')
df['Parking'] = df['Parking'].map({True:1,False:0})
# print(df['Address_Encoder'])
# print(df['Parking'])
df['Warehouse']=df['Warehouse'].map({True:1 , False:0})
df['HasParkingElevatorWarehouse']= df['Elevator']+df['Parking']+df['Warehouse']
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)

IQR = Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# print(len(df))
df2 = df[(df['Price'] >= lower_bound) & (df['Price']<= upper_bound)]
# print(len(df))
X = df2[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Warehouse','Age',
       'Address_Encoded','HasParkingElevatorWarehouse']]
y = df2['Price']
m = pd.concat([X,y],axis=1)

fig , axis = plt.subplots(nrows = 1 , ncols = 2 , figsize = (14, 6))
sns.boxplot(m,x='Price',ax = axis[0])
axis[0].set_title("After Remove Outlier Data")
sns.boxplot(df , x= 'Price' , ax = axis[1])
axis[1].set_title('Befor remove outlier data')
plt.tight_layout()
# plt.show()
sns.heatmap(data=m.corr()  ,annot=True)
# plt.show()
# print(X.head(10).to_string())
# print(y)
# print(df.dtypes)
# print(df.columns)
ss = StandardScaler()
x_Scaled = ss.fit_transform(X=X)
# print(x_Scaled)
# mms = MinMaxScaler()
# x_scaled_mms = mms.fit_transform(X)


x_train,x_test,y_train , y_test = train_test_split(x_Scaled,y ,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pridict = lr.predict(x_test)
# print(y_pridict - y_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,y_pridict)
MSE = mean_squared_error(y_test,y_pridict)
RMSE = np.sqrt(MSE)
print(MAE)
print(MSE)
# formated = f"{RMSE:,.0f}"
print(f"{RMSE:,.0f}")


