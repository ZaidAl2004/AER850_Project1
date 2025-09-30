# -*- coding: utf-8 -*-
"""
AER 850 Project 1
By: Zaid Al-lababidi
501176747
"""

"""
Importing Libraries
"""
import pandas as pd
import matplotlib.pyplot as mplot
import numpy as np
import seaborn as sb
import joblib as jl

"""
Step 1: Import Library
"""
data = pd.read_csv("C:\Github\AER850_Project1\Project 1 Data.csv")

print(data.head())

"""
Step 2: Perform Statistical Analysis
"""
fig, ax = mplot.subplots(subplot_kw={"projection":"3d"})
ax.plot(data["X"], data["Y"], data["Z"])
mplot.title("Statistical Representation of Data in 3d Space")

fig, ay = mplot.subplots()
ay.plot(data["Step"], data["X"], label='X')
ay.plot(data["Step"], data["Y"], label='Y')
ay.plot(data["Step"], data["Z"], label='Z')
mplot.legend()

fig, az = mplot.subplots()
az.plot(data["X"], label='X')
az.plot(data["Y"], label='Y')
az.plot(data["Z"], label='Z')
az.plot(data["Step"], label='Step')
mplot.legend()

fig, aw = mplot.subplots()
aw.scatter(data["Step"], data["X"], label='X')
aw.scatter(data["Step"], data["Y"], label='Y')
aw.scatter(data["Step"], data["Z"], label='Z')
mplot.legend()

"""
Step 3: Correlation Analysis
"""

x_mean = np.mean(data["X"])
y_mean = np.mean(data["Y"])
z_mean = np.mean(data["Z"])

xy_arr = []
xz_arr = []
yz_arr = []
x_arr = []
y_arr = []
z_arr = []

for i in range(1, 860):
    xy_arr.append((data["X"][i] - x_mean)*(data["Y"][i] - y_mean))
    xz_arr.append((data["X"][i] - x_mean)*(data["Z"][i] - z_mean))
    yz_arr.append((data["Y"][i] - y_mean)*(data["Z"][i] - z_mean))
    
    x_arr.append((data["X"][i] - x_mean)**2)
    y_arr.append((data["Y"][i] - y_mean)**2)
    z_arr.append((data["Z"][i] - z_mean)**2)
    
r_xy = sum(xy_arr)/np.sqrt(sum(x_arr)*sum(y_arr))
r_xz = sum(xz_arr)/np.sqrt(sum(x_arr)*sum(z_arr))
r_yz = sum(yz_arr)/np.sqrt(sum(y_arr)*sum(z_arr))

"""
Step 4: Classification Model Development/Engineering
"""

"""
Step 5: Model Performance Analysis
"""

"""
Step 6: Stacked Model Performance Analysis
"""

"""
Step 7: Model Evaluations
"""