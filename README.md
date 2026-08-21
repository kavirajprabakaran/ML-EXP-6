# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.

2. Split the dataset into training data and testing data.

3. Create and train the Logistic Regression model using the training data.

4. Predict the output values for the test data using the trained model.

5. Evaluate the model performance using accuracy score, classification report, and confusion matrix.


## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KAVIRAJ.P
RegisterNumber:  212225230135
```
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nclassification report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

## Output:
<img width="815" height="481" alt="image" src="https://github.com/user-attachments/assets/81f290ef-503d-43fc-a912-1bc6082f3000" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

