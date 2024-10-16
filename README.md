# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas as pd and import the required dataset.
2. Calculate the null values in the dataset.
3. Import the LabelEncoder from sklearn.preprocessing
4. Convert the string values to numeric values.
5. Import train_test_split from sklearn.model_selection.
6. Assign the train and test dataset.
7. Import DecisionTreeRegressor from sklearn.tree.
8. Import metrics from sklearn.metrics.
9. Calculate the MeanSquareError.
10. Apply the metrics to the dataset.
11. Predict the output for the required values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRAVESH N
RegisterNumber:  212223230154
*/
```
```python
import pandas as pd
data = pd.read_csv('/content/Salary (2).csv')

print(data.head())
print(data.info())
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])
print(data.head())

x = data[['Position','Level']]
y = data['Salary']

print("\n\n",x)

print("\n\n",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print("\n\n",x_test)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print(y_pred)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
print("mse : ",mse)

r2 = metrics.r2_score(y_test, y_pred)
print("\nr2 score : ",r2)
dt.predict([[5,6]])
```

## Output:
![Screenshot 2024-10-16 104734](https://github.com/user-attachments/assets/34b2c436-6c5c-446d-8cf3-c7cd0c8a8821)

![Screenshot 2024-10-16 104811](https://github.com/user-attachments/assets/247d6e4a-65a3-43b6-ab72-d58c8b7cfb34)

![Screenshot 2024-10-16 104850](https://github.com/user-attachments/assets/9955eaf4-ee9b-4842-8890-1c5d494d125e)

![Screenshot 2024-10-16 104927](https://github.com/user-attachments/assets/1dc04e17-e040-4d40-9d37-200af6c28220)

![Screenshot 2024-10-16 104951](https://github.com/user-attachments/assets/31e1178c-3f13-4ca2-903f-f2f68c5a2a02)

![Screenshot 2024-10-16 105001](https://github.com/user-attachments/assets/8037685a-7a53-46f0-90fa-fd6b3e72286a)

![Screenshot 2024-10-16 105017](https://github.com/user-attachments/assets/f085b97b-35c9-4138-b1a9-3c3e3356ae8c)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
