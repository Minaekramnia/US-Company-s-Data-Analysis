#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 20:41:55 2019

@author: minaekramnia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
import re
from sklearn.linear_model import LinearRegression

file1 = "/users/minaekramnia/Downloads/us_company.xlsx"
file2 = "/users/minaekramnia/Downloads/YahooFin.xlsx"
df1 = pd.read_excel(io=file1)
df2 = pd.read_excel(io=file2)

#Extracting number from MarketCap column text
def TextToString(text): 
    if ((type(text)==float) or (type(text)==int)):
        return text
    elif 'M' in text:    
        return float(re.sub('[^\d\.]', "", text))
    elif 'B' in text: 
        return 1000*float(re.sub('[^\d\.]', "", text))
    else:
        return float(re.sub('[^\d\.]', "", text))

#Question 1: Visualize the employee count

#Getting an idea of the data
df1[df1.EMPTOTAL>2000].shape

df1.hist(column = 'EMPTOTAL', bins=40,range=(0,2000))

#logarithmic scale
fig, ax = plt.subplots(figsize=(3, 3))
logEmp = np.log10(df1.EMPTOTAL)
ax = logEmp.hist(bins=40)
ax.set_xlabel("Log of Number of Employees",fontsize=10)
ax.set_ylabel("Frequency",fontsize=10)

fig, ax = plt.subplots(figsize=(10, 10))
logEmp1= np.log10(df2.Num_Employees)
ax = logEmp1.hist(bins=40)
ax.set_xlabel("Log of Number of Employees",fontsize=10)
ax.set_ylabel("Frequency",fontsize=10)

#Question 2: 
#Merge two datasets 
df3 = df1.merge(df2[['COMPANY-NAME','Num_Employees']],how='left',on='COMPANY-NAME')
df4 = df2.groupby('COMPANY-NAME').Num_Employees.first().reset_index() #getting the first of the repeated ones
df3 = df1.merge(df4,how='left',on='COMPANY-NAME')
mask = ~df3.Num_Employees.isnull() #excluding the NANs

#defining new dataframe that contains both of the datasets number of employees
dfcomp = df3[mask][['EMPTOTAL','Num_Employees']]
dfcomp.columns = ['emp1','emp2']

# Defining a new column difference, and calculating the difference betwen two
df3['difference'] = np.abs(df3.EMPTOTAL-df3.Num_Employees)
df3[df3.difference<5][['COMPANY-NAME','difference']]
#Method 1: First/Last ten companies
first_ten= df3.sort_values(by='difference', ascending=True)[0:10]
last_ten = df3.sort_values(by='difference', ascending=False)

# Correlation between Standard Deviation vs. Employee number
std_ind= df1.groupby('industry').EMPTOTAL.std()
df1.groupby('industry').EMPTOTAL.sum()
EMP_ind=df1.groupby('industry').EMPTOTAL.sum()
print(np.corrcoef(EMP_ind, std_ind)[1,0])

#Method 2: Percent Difference
df3['diff_perc']=df3.difference/df3.Num_Employees
df3[mask].diff_perc.sort_values().mean() 
#Method 2: First/Last ten companies
first_ten_perc= df3.iloc[df3[mask].diff_perc.sort_values().head(10).index]['COMPANY-NAME']
last_ten_perc=df3.iloc[df3[mask].diff_perc.sort_values().tail(10).index]['COMPANY-NAME']

#Calculating Root Mean Square Error or Cost Function
RMSD = np.sqrt(sum((dfcomp.emp1-dfcomp.emp2)**2))/359

# t-test 
stats.ttest_ind(dfcomp.emp1,dfcomp.emp2, equal_var = False)

#Question 3: 
fig, ax = plt.subplots(figsize=(4, 4))
totalEmp= df3.EMPTOTAL.sum()
(df1.groupby('industry').EMPTOTAL.sum()/totalEmp).plot('bar', ax = ax)
ax.set_ylabel("Total Employees",fontsize=10)

fig, ax = plt.subplots(figsize=(4, 4))
df1.groupby('industry').USSALES.sum().plot(kind= 'bar', ax = ax)
ax.set_ylabel("Total US Sales",fontsize=10)

fig, ax = plt.subplots(figsize=(10, 10))
df1['salesperlabor']= df1.USSALES/df1.EMPTOTAL
df1.groupby('industry').salesperlabor.mean().plot('bar', ax = ax)
plt.ylabel("Sales per Labor",fontsize=10)

#Plot correlations between USSALES and EMPTOTAL
fig, ax = plt.subplots(figsize=(4, 4))
mask = (~df1.EMPTOTAL.isnull()) & (df1.EMPTOTAL<250000) & (df1.USSALES<300000) 
x = df1[mask].EMPTOTAL[:, np.newaxis]
y= df1[mask].USSALES
model = LinearRegression(fit_intercept = True)
model.fit(x, y)
b = model.intercept_
m =model.coef_
f = lambda x: m*x + b
plt.plot(x,f(x), color = "orange", label= "fit line between min and max")
plt.scatter(df1[mask].EMPTOTAL, df1[mask].USSALES)
ax.set_xlabel("Number of Employees")
ax.set_ylabel("US SALES")
plt.gca().legend(('m = fitted line','Data'))
plt.show()

#df1[df1.industry=='Financial Services'].groupby('vertical').salesperlabor.mean().plot('pie')
#plt.ylabel("Sales per Labor",fontsize=10)

#Market Cap _ Industry - Call Function 
text = df2.MarketCap
df2['MarketCap2']=df2.apply(lambda row: TextToString(row['MarketCap']),axis=1)
df2.groupby('Sector').MarketCap2.sum().plot('bar')
plt.ylabel("Avg MarketCap (M)",fontsize=10)

# information regarding number of companies in each stock exchange
df2[df2.NASDAQ].Sector.value_counts().plot('bar')
df2[df2.AMEX].Sector.value_counts().plot('bar')
df2[df2.NYSE].Sector.value_counts().plot('bar')


mask = (~df1.EMPTOTAL.isnull()) & (df1.EMPTOTAL<250000)  
std_ind= df1.groupby('industry').EMPTOTAL.std()
df1.groupby('industry').EMPTOTAL.sum()
EMP_ind=df1.groupby('industry').EMPTOTAL.sum()
print(np.corrcoef(EMP_ind, std_ind)[1,0])

x = std_ind.values.reshape(-1,1)
y= df1.groupby('industry').EMPTOTAL.sum().values.reshape(-1,1)
model = LinearRegression(fit_intercept = True)
model.fit(x, y)
b = model.intercept_
m =model.coef_
f = lambda x: m*x + b
plt.plot(x,f(x), color = "orange", label= "fit line between min and max")
plt.scatter(x, y)
plt.xlabel("Number of Employees")
plt.ylabel("US SALES")
plt.gca().legend(('m = fitted line','Data'))
plt.show()
