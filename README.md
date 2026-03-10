# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select features and target variable.
2. Split the data into training and testing sets and apply feature scaling.
3. Train the SVM model using GridSearchCV to find the best parameters.
4. Predict the test data and evaluate the model using accuracy and classification report.
5. Generate and display the confusion matrix to visualize model performance. 

## Program:
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat','Sugars', 'Dietary Fiber', 'Protein']
target='class'
X=data[features]
y=data[target]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC()
param_grid={
    'C':[0.1, 1, 10, 100],
    'kernel':['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search=GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model=grid_search.best_estimator_
print("Name: SARAN SADASIVAM")
print("Register Number: 212225040385")
print("Best Parameters:",grid_search.best_params_)
y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Name: SARAN SADASIVAM")
print("Register Number: 212225040385")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test, y_pred))
conf_matrix=confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="769" height="603" alt="image" src="https://github.com/user-attachments/assets/adea927e-08d8-4169-a910-39a35e35e2b3" />

<img width="613" height="345" alt="image" src="https://github.com/user-attachments/assets/df3e0486-8ccf-4b89-ba14-789f62f27f2e" />


<img width="720" height="495" alt="image" src="https://github.com/user-attachments/assets/a301110b-7ad2-4cc6-a936-5e0e80a11676" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
