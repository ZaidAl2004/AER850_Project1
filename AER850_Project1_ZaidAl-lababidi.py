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

"""
Step 2: Perform Statistical Analysis
"""
fig, ax = mplot.subplots(subplot_kw={"projection":"3d"})
ax.plot(data["X"], data["Y"], data["Z"])
mplot.title("Statistical Representation of Data in 3d Space")

grouped_data = data.groupby('Step')

fig, step_split = mplot.subplots(subplot_kw={"projection":"3d"})
mplot.title("Statistical Representation of Steps Split by Count in 3d Space")

for i in list(range(1,14)):
    step_split.plot(grouped_data.get_group(i)["X"], grouped_data.get_group(i)["Y"], grouped_data.get_group(i)["Z"])

#Uncomment following line for legend (couldn't take the overlap off)
#step_split.legend(['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Step 7', 'Step 8', 'Step 9', 'Step 10', 'Step 11', 'Step 12', 'Step 13'], loc='best')

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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

data["Steps Split"] = pd.cut(data["Step"], bins = (list(range(0, 14))), labels = list(range(1,14)))
splitter_tool = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter_tool.split(data, data["Steps Split"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["Steps Split"], axis=1)
strat_data_test = strat_data_test.drop(columns=["Steps Split"], axis = 1)

#Creates a test and train list for the coordinates for each steps
step_train = strat_data_train["Step"]
coord_train = strat_data_train.drop(columns=["Step"])
step_test = strat_data_test["Step"]
coord_test = strat_data_test.drop(columns=["Step"]) 

corr_matrix = strat_data_train.corr()
f, bx = mplot.subplots(figsize=(11, 9))
cmap = sb.diverging_palette(230, 20, as_cmap=True)
sb.heatmap(corr_matrix, cmap=cmap)


#Data Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(coord_train)

#pd.DataFrame(coord_train).to_csv("UnscaledOriginalData.csv")
coord_train = sc.transform(coord_train)
#pd.DataFrame(coord_train).to_csv("ScaledOriginalData.csv")

coord_test = sc.transform(coord_test)

#Developing the first model, Linear Regression which will be used to as a baseline to evaluate the other models.
from sklearn.linear_model import LinearRegression
mdl1 = LinearRegression()
mdl1.fit(coord_train, step_train)
mdl1.fit(coord_test, step_test)

step_pred_train1 = mdl1.predict(coord_train)
for i in range(5):
    print("Predictions: ", step_pred_train1[i], "Actual Values: ", step_train[i])

print("Training accuracy:", mdl1.score(coord_train, step_train))
print("Test accuracy:", mdl1.score(coord_test, step_test))

#Developing the second model SVC (Support Vector Classifier)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
mdl2 = Pipeline([("Scaler", sc), ("clf", SVC(kernel="linear", probability=True, random_state=42))])
mdl2.fit(coord_train, step_train)
step_pred_train2=mdl2.predict(coord_train)
for i in range(5):
    print("Predictions: ", step_pred_train2[i], "Actual Values: ", step_train[i])
print("SVC Training Accuracy", mdl2.score(coord_train, step_train))
print("SVC Testing Accuracy", mdl2.score(coord_test, step_test))

#Last model used will be decision tree. This model will be used as it has a certain ability to tune its depth.
from sklearn.tree import DecisionTreeClassifier
mdl3 = DecisionTreeClassifier(max_depth=10, random_state=42)
mdl3.fit(coord_train, step_train)
print("Decision Tree Training Accuracy: ", mdl2.score(coord_train, step_train))
print("Decision Tree Test Accuracy: ", mdl2.score(coord_test, step_test))


"""
Step 5: Model Performance Analysis
"""

"""
Step 6: Stacked Model Performance Analysis
"""

"""
Step 7: Model Evaluations
"""
