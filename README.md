# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/

import pandas as pd

data = pd.read_csv("/Employee_EX6 (2).csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion="entropy") 
dt.fit(x_train, y_train) 
y_pred = dt.predict(x_test)

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test, y_pred)

accuracy

print(accuracy)

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

```

## Output:

### 1.DATA:
![image](https://github.com/user-attachments/assets/3c059c86-3535-4707-8e84-428ab95cd8ea)

### 2.ACCURACY:
![image](https://github.com/user-attachments/assets/12ea4ecb-0d53-48d5-8626-2d78486cf770)

### 3.PREDICTION:
![image](https://github.com/user-attachments/assets/1ad9c07e-3a13-42d5-9c04-52a25e7f37ab)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
