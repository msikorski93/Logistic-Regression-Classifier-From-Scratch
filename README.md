# Logistic-Regression-Classifier-From-Scratch

![ alt text ](https://img.shields.io/badge/license-MIT-green?style=&logo=)
![ alt text ](https://img.shields.io/badge/-Python-3776AB?logo=Python&logoColor=white)
![ alt text ](https://img.shields.io/badge/-Jupyter-F37626?logo=Jupyter&logoColor=white)
![ alt text ](https://img.shields.io/badge/-NumPy-013243?logo=Numpy&logoColor=white)
![ alt text ](https://img.shields.io/badge/-pandas-150458?logo=Pandas&logoColor=white)

Logistic regression with the ability of implementing penalty terms. The class object is dedicated to binary classification and is compatible with `scikit-learn`'s GridSearchCV method.<br>
An example of its usage was performed on the WBCD dataset and returned the following estimates:
* Accuracy: 0.9883
* Recall: 0.9841
* Precision: 0.9841
* Error Rate: 0.0117
* F1 Score: 0.9841
* ROC: 0.9874
* Specificity: 0.9907
* Misclassified Samples: 2 (out of 171)

<img src='https://github.com/user-attachments/assets/a1f2f302-fcc9-42ba-91b2-2e986c608742' height='300'/>

**Code example:**
```
from logistic_regression import LogisticRegression

log_model = LogisticRegression(learning_rate=0.01, C=0.1, num_iter=20, penalty='elasticnet', l1_ratio=0.7)
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test, threshold=0.6)
```
