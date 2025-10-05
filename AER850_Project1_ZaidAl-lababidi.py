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

"""
Step 1: Import Library
"""
data = pd.read_csv("C:\Github\AER850_Project1\Project 1 Data.csv")

"""
Step 2: Perform Statistical Analysis
"""
grouped_data = data.groupby('Step')

fig, step_split = mplot.subplots(subplot_kw={"projection":"3d"})
mplot.title("Statistical Representation of Steps Split by Count in 3d Space")

for i in list(range(1,14)):
    step_split.plot(grouped_data.get_group(i)["X"], grouped_data.get_group(i)["Y"], grouped_data.get_group(i)["Z"])

#Uncomment following line for legend (couldn't take the overlap off)
step_split.legend(['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Step 7', 'Step 8', 'Step 9', 'Step 10', 'Step 11', 'Step 12', 'Step 13'], loc='best')

"""
Step 3: Correlation Analysis
"""
from sklearn.model_selection import StratifiedShuffleSplit

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


"""
Step 4: Classification Model Development/Engineering
"""
#Data Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(coord_train)

#Developing the first model, Logistic Regression which will be used to as a baseline to evaluate the other models.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
mdl1 = Pipeline([("scaler", StandardScaler()),("clf", LogisticRegression())])
mdl1.fit(coord_train, step_train)
mdl1.fit(coord_test, step_test)

step_pred_train1 = mdl1.predict(coord_train)
for i in range(5):
    print("Predictions: ", step_pred_train1[i], "Actual Values: ", step_train[i])

from sklearn.model_selection import GridSearchCV
param1 = {
    "clf__C": [1, 10, 100, 1000]
    }
gs1=GridSearchCV(mdl1, param_grid=param1, scoring="roc_auc", cv=5, n_jobs=-1)
gs1.fit(coord_train, step_train)
#Updating first model to best estimator
mdl1 = gs1.best_estimator_

#Developing the second model SVC (Support Vector Classifier)
from sklearn.svm import SVC
mdl2 = Pipeline([("Scaler", sc), ("clf", SVC(kernel="linear", probability=True, random_state=42))])
mdl2.fit(coord_train, step_train)
step_pred_train2=mdl2.predict(coord_train)
for i in range(5):
    print("Predictions: ", step_pred_train2[i], "Actual Values: ", step_train[i])

#Improving SVC model using Grid Search
param2= {
    "clf__kernel": ["linear", "rbf"],
    "clf__C": [1, 10, 100, 1000],
    "clf__gamma": ["scale"]
    }
gs2=GridSearchCV(mdl2, param_grid=param2, scoring="roc_auc", cv=5, n_jobs=-1)
gs2.fit(coord_train, step_train)
#updating second model to best estimator
mdl2 = gs2.best_estimator_

#Last model used will be decision tree.
from sklearn.tree import DecisionTreeClassifier
mdl3 = DecisionTreeClassifier(max_depth=10, random_state=42)
mdl3.fit(coord_train, step_train)

#Improving Decision Tree classifier using Grid Search
param3= {
    "max_depth": [None, 6, 10, 16]
    }
gs3=GridSearchCV(mdl3, param_grid=param3, scoring="roc_auc", cv=5, n_jobs=-1)
gs3.fit(coord_train, step_train)
#updating third model to best estimator
mdl3 = gs3.best_estimator_

#Logistic Regression using randomized search cv
from sklearn.model_selection import RandomizedSearchCV
param4 = {
    'clf__C': [0.01, 0.1, 1, 10, 100, 1000]
    }
mdl1_rscv = RandomizedSearchCV(mdl1, param4, random_state=42)
search = mdl1_rscv.fit(coord_train, step_train)
print("The best parameters are:", search.best_params_)


"""
Step 5: Model Performance Analysis
"""

#Model Performance Analysis for model 1 (Logistic Regression)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print("Logistic Regression Training accuracy:", mdl1.score(coord_train, step_train))
print("Logistic Regression Test accuracy:", mdl1.score(coord_test, step_test))

#Confusion Matrix
step_pred_mdl1 = mdl1.predict(coord_test)
cm_mdl1 = confusion_matrix(step_test, step_pred_mdl1)
print("Confusion Matrix of Logistic Regression Model:")
print(cm_mdl1)

#Showing precision, recall and f1 scores
precision_score_mdl1 = precision_score(step_test, step_pred_mdl1, average="micro")
recall_score_mdl1 = recall_score(step_test, step_pred_mdl1, average="micro")
f1_score_mdl1 = f1_score(step_test, step_pred_mdl1, average="micro")
print("Precision of Logistic Regression: ", precision_score_mdl1)
print("Recall of Logistic Regression: ", recall_score_mdl1)
print("F1 Score of Logistic Regression: ", f1_score_mdl1)


#Model Performance Analysis for model 2 (Support Vector Classifier)
print("SVC Training Accuracy", mdl2.score(coord_train, step_train))
print("SVC Testing Accuracy", mdl2.score(coord_test, step_test))

#Confusion Matrix
step_pred_mdl2 = mdl2.predict(coord_test)
cm_mdl2 = confusion_matrix(step_test, step_pred_mdl2)
print("Confusion Matrix of SVC Model:")
print(cm_mdl2)

#Showing precision, recall and f1 scores
precision_score_mdl2 = precision_score(step_test, step_pred_mdl2, average="weighted")
recall_score_mdl2 = recall_score(step_test, step_pred_mdl2, average="micro")
f1_score_mdl2 = f1_score(step_test, step_pred_mdl2, average="micro")
print("Precision of SVC Model: ", precision_score_mdl2)
print("Recall of SVC Model: ", recall_score_mdl2)
print("F1 Score of SVC Model: ", f1_score_mdl2)


#Model Performance Analysis for model 3 (Decision Tree)
print("Decision Tree Training Accuracy: ", mdl2.score(coord_train, step_train))
print("Decision Tree Test Accuracy: ", mdl2.score(coord_test, step_test))

#Confusion Matrix
step_pred_mdl3 = mdl3.predict(coord_test)
cm_mdl3 = confusion_matrix(step_test, step_pred_mdl3)
print("Confusion Matrix of Decision Tree:")
print(cm_mdl3)

#Showing precision, recall and f1 scores
precision_score_mdl3 = precision_score(step_test, step_pred_mdl3, average="micro")
recall_score_mdl3 = recall_score(step_test, step_pred_mdl3, average="micro")
f1_score_mdl3 = f1_score(step_test, step_pred_mdl3, average="micro")
print("Precision of Decision Tree: ", precision_score_mdl3)
print("Recall of Decision Tree: ", recall_score_mdl3)
print("F1 Score of Decision Tree: ", f1_score_mdl3)


"""
Step 6: Stacked Model Performance Analysis
"""

from sklearn.ensemble import StackingClassifier
estimators = [('dt', mdl3), ('svr', mdl2)]
comb_mdl = StackingClassifier(estimators=estimators, final_estimator=mdl1)
comb_mdl_fitted_train = comb_mdl.fit(coord_train, step_train)
comb_mdl_pred = comb_mdl.predict(coord_test)

print("Combined Model Training Accuracy: ", comb_mdl.score(coord_train, step_train))
print("Combined Model Test Accuracy: ", comb_mdl.score(coord_test, step_test))

cm_comb_mdl = confusion_matrix(step_test, comb_mdl_pred)
print("Confusion Matrix of Combined Model: ")
print(cm_comb_mdl)

precision_score_comb_mdl = precision_score(step_test, comb_mdl_pred, average="micro")
recall_score_comb_mdl = recall_score(step_test, comb_mdl_pred, average="micro")
f1_score_comb_mdl = f1_score(step_test, comb_mdl_pred, average="micro")
print("Precision of Combined Model: ", precision_score_comb_mdl)
print("Recall of Combined Model: ", recall_score_comb_mdl)
print("F1 Score of Combined Model: ", f1_score_comb_mdl)


"""
Step 7: Model Evaluations
"""

import joblib as jl
jl.dump(comb_mdl, "combined_model.pkl")
loaded_mdl = jl.load("combined_model.pkl")
test_array = ([9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3])
loaded_mdl_pred = loaded_mdl.predict(test_array)

mdl1_test_pred = mdl1.predict(test_array)
mdl2_test_pred = mdl2.predict(test_array)
mdl3_test_pred = mdl3.predict(test_array)
print(loaded_mdl_pred)
print(mdl1_test_pred)
print(mdl2_test_pred)
print(mdl3_test_pred)