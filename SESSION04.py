import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import missingno
#  CRISP
#  bussiness understanding
#  Data understanding
# Data Preparation
#  Data Modeling
# Deployment
#  List , Dictionary, Set (Mutable),
#  Tuple , string , int / float / boolean (Immutable)
list1 = [1,2,3]
Set1 = ("a","b","C")
dic1 = {"FulName" : ["Ali","Mohammad","Roza","Sahar"]
    , "Age" : [18,25,22,30]}
df1 = pd.DataFrame(dic1)
# print(df1)


df = pd.read_csv('Advertising.csv',usecols=['TV','radio','newspaper','sales'])
# print(df.to_string())
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
# print(df.isna().sum())      #print(df.isnull.sum())
# print('--------------------- head ------------)')
# print(df.head(10).to_string())
# print('--------------------- tail() ------------)')
# print(df.tail(15).to_string())
missingno.matrix(df)
plt.show()
