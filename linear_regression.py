# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:48:01 2016

@author: AbreuLastra_Work
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm


loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

# Visualize Interest rate
print(loansData['Interest.Rate'][0:5])

# Removing '%' from the variable

# First a lambda function to remove characters


loansData['clean_Interest_rate'] = map (lambda x:round(float(x.rstrip('%')) / 100, 4), loansData['Interest.Rate'])

# Vizualize Loan.Length
print(loansData['Loan.Length'][0:5])

# Removing ' months' from the variale

loansData['clean_Loan_length'] = map (lambda y: float (y.strip(' months')), loansData['Loan.Length'])
print(loansData['clean_Loan_length'][0:5])

#visualize FICO.Range
print(loansData['FICO.Range'][0:5])

#Remove low score in range

loansData['FICO_Score'] = map (lambda z:float(z.split("-")[1]), loansData['FICO.Range'])

print(loansData['FICO_Score'][0:5])


#Plotting a histogram for interest rate data
plt.figure()
p = loansData['clean_Interest_rate'].hist()
plt.show()

#Plotting loan lenght data
plt.figure()
p = loansData['clean_Loan_length'].hist()
plt.show()

#Plotting a histogram for FICO Score data
plt.figure()
p = loansData['FICO_Score'].hist()
plt.show()

# Creating a scatterplot matrix
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))

# To run regressions
intrate = loansData['clean_Interest_rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO_Score']

# Shaping the dependent variable
y = np.matrix(intrate).transpose()
# the independent variable are columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# This puts the dependent variables in a matrix
x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

#this gives you the output
f.summary()
plt.plot(intrate, fico, "x")
plt.show()
